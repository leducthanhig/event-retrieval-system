import { useState } from 'react';
import Home from './pages/Home';
import './styles/layout.css';
import useSearch from './hooks/useSearch';

export default function App() {
  // Core search states
  const [query, setQuery] = useState('');
  const [selectedModels, setSelectedModels] = useState(['siglip2']);
  const [weights, setWeights] = useState({ siglip2: '1.0', siglip: '0.0', quickgelu: '0.0' });

  // Search control states
  const [isSearching, setIsSearching] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);

  // Results & selection
  const [results, setResults] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null); // for preview modal/video

  // API hooks
  const { search, rewrite, loading, error, setError } = useSearch({
    onResults: (items) => setResults(items || []),
  });

  const handleSearch = async () => {
    // Validate query
    const q = query.trim();
    if (!q) {
      setError('Please enter a query.');
      return;
    }

    const n = selectedModels.length;
    if (n === 0) { 
      setError('Please select at least one model.'); 
      return; 
    }

    // Validate weights in multi-model mode
    if (n >= 2) {
      const ws = selectedModels.map(k => parseFloat(weights[k]));
      if (ws.some(x => Number.isNaN(x))) { 
        setError('Please enter valid decimal weights.'); 
        return; 
      }
      const sum = ws.reduce((a,b) => a + b, 0);
      const roundedSum = Math.round(sum * 10) / 10;
      if (roundedSum !== 1.0) { 
        setError('Weights must sum exactly to 1.'); 
        return; 
      }
    }

    // map UI keys -> backend models
    const MODEL_MAP = {
      siglip2:   { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
      siglip:    { name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
      quickgelu: { name: 'ViT-L-14-quickgelu',  pretrained: 'dfn2b'  },
    };

    setError('');
    setIsSearching(true);

    try {
      let body;
      if (n === 1) {
        body = {
          models: MODEL_MAP[selectedModels[0]],
        };
      } else {
        const models = selectedModels.map(k => MODEL_MAP[k]);
        const ws = selectedModels.map(k => Number(weights[k]));
        body = {
          models,
          weights: ws,
        };
      }

      // Build URL with query params (q, top, pooling_method)
      const url = `http://localhost:8000/search?q=${encodeURIComponent(q)}&top=100`;

      // Call API
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);

    } catch (e) {
        console.error('Search error:', e);
        setError('Error calling API: ' + e.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleRewrite = async () => {
    setError('');
    setIsRewriting(true);
    try {
      const MODEL_MAP = {
      siglip2: { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
      siglip:  { name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
      quickgelu: { name: 'ViT-L-14-quickgelu', pretrained: 'dfn2b' },
    };
    const first = selectedModels[0];
    const clip_model = first ? MODEL_MAP[first] : undefined;
    const data = await rewrite({
      text: query,
      model: 'gemini-2.5-flash-lite',
      clip_model,
      thinking: false,
    });
     if (data?.rewritten_query) {
        setQuery(data.rewritten_query);
     }
    } finally {
      setIsRewriting(false);
    }
  };

  // Unified search, any combination of [text, image, metadata]
  const handleUnifiedSearch = async ({ modes, imageFile, metadataQuery, modalityWeights } = {}) => {
    const MODEL_MAP = {
      siglip2:   { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
      siglip:    { name: 'ViT-L-16-SigLIP-256',   pretrained: 'webli' },
      quickgelu: { name: 'ViT-L-14-quickgelu',    pretrained: 'dfn2b' },
    };

    // Fallback: if modes is missing, default to text-only
    const resolvedModes = modes ?? { text: true, image: false, metadata: false };

    const n = selectedModels.length;
    const modelsPayload = (n === 1)
      ? MODEL_MAP[selectedModels[0]]
      : selectedModels.map(k => MODEL_MAP[k]);

    setError('');
    setIsSearching(true);
    try {
      const fd = new FormData(); // always multipart (image may be present)

      // TEXT branch
      if (resolvedModes.text) {
        const q = (query || '').trim();
        if (!q) {
          setIsSearching(false);
          setError('Please enter a query.');
          return;
        }
        fd.append('text_query', q);
        fd.append('models', JSON.stringify(modelsPayload));
        if (n >= 2) {
          const ws = selectedModels.map(k => Number(weights[k]));
          const rounded = Math.round(ws.reduce((a,b)=>a+b,0) * 10) / 10;
          if (Number.isNaN(rounded) || rounded !== 1.0) {
            setIsSearching(false);
            setError('Weights must sum exactly to 1.');
            return;
          }
          fd.append('model_weights', JSON.stringify(ws));
        }
      }

      // IMAGE branch
      if (resolvedModes.image && imageFile) {
        fd.append('image_query', imageFile);
      }

      // METADATA branch (reuse same query)
      if (resolvedModes.metadata) {
        const mq = (metadataQuery || '').trim();
        if (mq) fd.append('metadata_query', mq);
      }

      // Modality weights (optional)
      if (modalityWeights) {
        fd.append('modality_weights', JSON.stringify(modalityWeights));
      }

      // Pooling & top
      fd.append('pooling_method', 'max');

      const url = `http://localhost:8000/search_unified?top=100`;
      const res = await fetch(url, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Search failed with status ${res.status}`);
      const data = await res.json();
      setResults(data?.results || []);
    } catch (e) {
      console.error('Unified search error:', e);
      setError('Error calling API: ' + (e?.message || 'unknown'));
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <Home
      query={query}
      setQuery={setQuery}
      selectedModels={selectedModels}
      setSelectedModels={setSelectedModels}
      weights={weights}
      setWeights={setWeights}
      results={results}
      setSelectedItem={setSelectedItem}
      onSearch={handleSearch}
      onRewrite={handleRewrite}
      onUnifiedSearch={handleUnifiedSearch}
      loading={loading}
      isSearching={isSearching}
      isRewriting={isRewriting}
      error={error}
      selectedItem={selectedItem}
    />
  );
}