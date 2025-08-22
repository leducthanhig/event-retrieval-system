import { useState, useEffect } from 'react';
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

  // Weights for different modalities
  const [modalityWeights, setModalityWeights] = useState({
    text: 1.0,
    image: 0.0,
    transcription: 0.0,
    metadata: 0.0,
  });

  // Active search modes (tabs)
  const [activeTabs, setActiveTabs] = useState(['text']);
  const toggleTab = (id) => {
    setActiveTabs(prev => {
      const on = prev.includes(id);
      const next = on ? prev.filter(t => t !== id) : [...prev, id];
      // sync weight về 0 khi tắt
      if (on) setModalityWeights(w => ({ ...w, [id]: 0.0 }));
      return next;
    });
  };
  
  useEffect(() => {
    setModalityWeights(w => {
      const next = { ...w };
      const keys = ['text','image','transcription','metadata'];
      keys.forEach(k => {
        if (!activeTabs.includes(k)) next[k] = 0.0;
      });
      return next;
    });
  }, [activeTabs, setModalityWeights]);

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
      modalityWeights: mwOverride 
    } = {}) => {

    // Fallback: if modes is missing, default to text-only
    const resolvedModes = modes ?? { text: true, image: false, transcription: false, metadata: false };

    const MODEL_MAP = {
      siglip2:   { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
      siglip:    { name: 'ViT-L-16-SigLIP-256',   pretrained: 'webli' },
      quickgelu: { name: 'ViT-L-14-quickgelu',    pretrained: 'dfn2b' },
    };

    const n = selectedModels.length;
    const modelsPayload = (n === 1)
      ? MODEL_MAP[selectedModels[0]]
      : selectedModels.map(k => MODEL_MAP[k]);

    setError('');
    setIsSearching(true);

    try {
      const fd = new FormData(); // always multipart

      // Validate search modes
      if (!resolvedModes.text && !resolvedModes.image && !resolvedModes.transcription && !resolvedModes.metadata) {
        return failSearch('Please select at least one search mode.');
      }

      // Validate modality weights
      const activeKeys = ['text', 'image', 'transcription', 'metadata'].filter(k => resolvedModes[k]);
      const mw = mwOverride ?? modalityWeights ?? {};
      if (activeKeys.length >= 2) {
        const ws = activeKeys.map(k => Number(mw?.[k] ?? 0));
        if (ws.some(x => Number.isNaN(x))) {
          return failSearch('Please enter valid decimal modality weights.');
        }
        const sum = ws.reduce((a,b)=>a+b, 0);
        const roundedSum = Math.round(sum * 10) / 10; // round to nearest 0.1
        if (roundedSum !== 1.0) {
          return failSearch('Modality weights must sum exactly to 1.');
        }
        // send attached weights to backend with same keynames
        fd.append(
          'modality_weights', 
          JSON.stringify(activeKeys.reduce((obj, k, i) => ((obj[k] = ws[i]), obj), {}))
        );
      } else if (activeKeys.length === 1) {
        // 1 mode: skip weights
      }

      // Validate top-k
      const k = parseInt(topK, 10);
      if (Number.isNaN(k) || k <= 0 || k > 500) {
        return failSearch('Top-k must be an integer between 1 and 500.');
      }

      // TEXT branch
      if (resolvedModes.text) {
        const q = (query || '').trim();
        if (!q) {
          return failSearch('Please enter a text query.');
        }
        fd.append('text_query', q);
        fd.append('models', JSON.stringify(modelsPayload));
        if (n >= 2) {
          const ws = selectedModels.map(k => Number(weights[k]));
          const rounded = Math.round(ws.reduce((a,b)=>a+b,0) * 10) / 10;
          if (Number.isNaN(rounded) || rounded !== 1.0) {
            return failSearch('Weights must sum exactly to 1.');
          }
        }
      }

      // IMAGE branch
      if (resolvedModes.image) {
        if (!imageFile) {
          return failSearch('Please upload an image.');
        }
        fd.append('image_query', imageFile);
      }

      // TRANSCRIPT branch
      if (resolvedModes.transcription) {
        const tq = (transcriptQuery || '').trim();
        if (!tq) {
          return failSearch('Please enter a transcription query.');
        }
        if (tq) fd.append('transcription_query', tq);
      }

      // METADATA branch
      if (resolvedModes.metadata) {
        const mq = (metadataQuery || '').trim();
        if (!mq) {
          return failSearch('Please enter a metadata query.');
        }
        if (mq) fd.append('metadata_query', mq);
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
      // validate query
      const q = (query || '').trim();
      if (!q) {
        return failSearch('Please enter a text query.');
      }

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
      activeTabs={activeTabs}
      setActiveTabs={setActiveTabs}
      modalityWeights={modalityWeights}
      setModalityWeights={setModalityWeights}
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