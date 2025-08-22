import { useState } from 'react';
import Home from './pages/Home';
import './styles/layout.css';
import useSearch from './hooks/useSearch';

export default function App() {
  // Core search states
  const [topK, setTopK] = useState('100');
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

  // Clear previous results and selection
  const clearResults = () => {
    setResults([]);
    setSelectedItem(null);
  };

  // Unified way to fail a search: set error, clear results, stop spinner
  const failSearch = (msg) => {
    setError(msg);
    clearResults();
    setIsSearching(false);
  };

  const handleSearch = async ({ 
      modes, 
      imageFile, 
      transcriptQuery, 
      metadataQuery, 
      modalityWeights 
    } = {}) => {
    const MODEL_MAP = {
      siglip2:   { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
      siglip:    { name: 'ViT-L-16-SigLIP-256',   pretrained: 'webli' },
      quickgelu: { name: 'ViT-L-14-quickgelu',    pretrained: 'dfn2b' },
    };

    // Fallback: if modes is missing, default to text-only
    const resolvedModes = modes ?? { text: true, image: false, transcript: false, metadata: false };

    const n = selectedModels.length;
    const modelsPayload = (n === 1)
      ? MODEL_MAP[selectedModels[0]]
      : selectedModels.map(k => MODEL_MAP[k]);

    setError('');
    setIsSearching(true);
    try {
      // Validate search modes
      if (!resolvedModes.text && !resolvedModes.image && !resolvedModes.transcription && !resolvedModes.metadata) {
        return failSearch('Please select at least one search mode.');
      }

      // Validate top-k
      const k = parseInt(topK, 10);
      if (Number.isNaN(k) || k <= 0 || k > 500) {
        setIsSearching(false);
        return failSearch('Top-k must be an integer between 1 and 500.');
      }

      const fd = new FormData(); // always multipart (image may be present)

      // TEXT branch
      if (resolvedModes.text) {
        const q = (query || '').trim();
        if (!q) {
          return failSearch('Please enter a query.');
        }
        fd.append('text_query', q);
        fd.append('models', JSON.stringify(modelsPayload));
        if (n >= 2) {
          const ws = selectedModels.map(k => Number(weights[k]));
          const rounded = Math.round(ws.reduce((a,b)=>a+b,0) * 10) / 10;
          if (Number.isNaN(rounded) || rounded !== 1.0) {
            return failSearch('Weights must sum exactly to 1.');
          }
          fd.append('model_weights', JSON.stringify(ws));
        }
      }

      // IMAGE branch
      if (resolvedModes.image && imageFile) {
        fd.append('image_query', imageFile);
      }

      // TRANSCRIPT branch
      if (resolvedModes.transcript) {
        const tq = (transcriptQuery || '').trim();
        if (tq) fd.append('transcription_query', tq);
      }

      // METADATA branch
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

      const url = `/search?top=${k || 100}`;
      const res = await fetch(url, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Search failed with status ${res.status}`);
      const data = await res.json();
      setResults(data?.results || []);
    } catch (e) {
      console.error('Unified search error:', e);
      clearResults();
      setError('Error calling API: ' + (e?.message || 'unknown'));
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

  return (
    <Home
      query={query}
      setQuery={setQuery}
      topK={topK}
      setTopK={setTopK}
      selectedModels={selectedModels}
      setSelectedModels={setSelectedModels}
      weights={weights}
      setWeights={setWeights}
      results={results}
      setSelectedItem={setSelectedItem}
      onSearch={handleSearch}
      onRewrite={handleRewrite}
      loading={loading}
      isSearching={isSearching}
      isRewriting={isRewriting}
      error={error}
      selectedItem={selectedItem}
      onSimilarSearch={(file) =>
        handleSearch({
          modes: { text: false, image: true, transcription: false, metadata: false },
          imageFile: file,
        })
      }
    />
  );
}