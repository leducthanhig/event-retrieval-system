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
  const [results, setResults] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null); // for preview modal/video
  const [temporalMode, setTemporalMode] = useState(false);
  const [modalityWeights, setModalityWeights] = useState({
    text: 1.0,
    image: 0.0,
    transcription: 0.0,
    metadata: 0.0,
  });

  // Map UI model keys to backend payload format
  const MODEL_MAP = {
    siglip2:   { name: 'ViT-B-16-SigLIP2-384', pretrained: 'webli' },
    siglip:    { name: 'ViT-L-16-SigLIP-256',   pretrained: 'webli' },
    quickgelu: { name: 'ViT-L-14-quickgelu',    pretrained: 'dfn2b' },
  };

  // Active search modes (tabs)
  const [activeTabs, setActiveTabs] = useState(['text']);
  const toggleTab = (id) => {
    setActiveTabs(prev => {
      const on = prev.includes(id);
      const next = on ? prev.filter(t => t !== id) : [...prev, id];
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

  // Session-scoped text search history
  const [searchHistory, setSearchHistory] = useState(() => {
    try {
      const raw = sessionStorage.getItem('text_search_history');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });

  // Persist to sessionStorage on change
  useEffect(() => {
    try {
      sessionStorage.setItem('text_search_history', JSON.stringify(searchHistory));
    } catch {}
  }, [searchHistory]);

  // Helper: push to history (dedupe head, max 50)
  const pushHistory = (text) => {
    const q = (text || '').trim();
    if (!q) return;
    setSearchHistory(prev => {
      if (prev.length && prev[0]?.text === q) return prev; // no immediate dup
      const next = [{ text: q, ts: Date.now() }, ...prev.filter(it => it?.text !== q)];
      return next.slice(0, 50);
    });
  };

  // API hooks
  const { search, rewrite, isSearching, isRewriting, error, setError } = useSearch({
    onResults: (items) => setResults(items || []),
  });

  // Clear previous results and selection
  const clearResults = () => {
    setResults([]);
    setSelectedItem(null);
    setTemporalMode(false);
  };

  // Unified way to fail a search: set error, clear results
  const failSearch = (msg) => {
    setError(msg);
    clearResults();
  };

  const onSearch = async ({
    modes,
    imageFile,
    transcriptQuery,
    metadataQuery,
    modalityWeights: mwOverride,
  } = {}) => {
    setError('');

    // Resolve active modes
    const resolvedModes = modes ?? { text: true, image: false, transcription: false, metadata: false };
    const activeKeys = ['text','image','transcription','metadata'].filter(k => resolvedModes[k]);
    if (activeKeys.length === 0) {
      return failSearch('Please select at least one search mode.');
    }

    // Validate Top-K
    const k = parseInt(topK, 10);
    if (Number.isNaN(k) || k <= 0 || k > 500) {
      return failSearch('Top-K must be an integer between 1 and 500.');
    }

    // Map selected models to backend payload format
    const n = selectedModels.length;
    const modelsPayload = (n === 1) 
      ? MODEL_MAP[selectedModels[0]] 
      : selectedModels.map(k => MODEL_MAP[k]);

    // Parse topK with safe fallback
    const top = parseInt(topK, 10) || 100;

    if (n >= 2) {
      const ws = selectedModels.map(k => Number(weights[k]));
      if (ws.some(x => Number.isNaN(x))) {
        return failSearch('Model weights must be decimals.');
      }
      const sum = ws.reduce((a,b)=>a+b, 0);
      const roundedSum = Math.round(sum * 10) / 10;
      if (roundedSum !== 1.0) {
        return failSearch('Model weights must sum exactly to 1.');
      }
    }

    // Validate per-modality inputs
    const q = (query || '').trim();
    if (resolvedModes.text) {
      if (!q) return failSearch('Please enter a text query.');
    }
    if (resolvedModes.image) {
      if (!imageFile) return failSearch('Please upload an image.');
    }
    if (resolvedModes.transcription) {
      const tq = (transcriptQuery || '').trim();
      if (!tq) return failSearch('Please enter a transcription query.');
    }
    if (resolvedModes.metadata) {
      const mq = (metadataQuery || '').trim();
      if (!mq) return failSearch('Please enter a metadata query.');
    }

    // Validate modality weights when combining >=2 modes
    const mw = mwOverride ?? modalityWeights ?? {};
    if (activeKeys.length >= 2) {
      const ws = activeKeys.map(k => Number(mw?.[k] ?? 0));
      if (ws.some(x => Number.isNaN(x))) {
        return failSearch('Please enter valid decimal modality weights.');
      }
      const sum = ws.reduce((a,b)=>a+b, 0);
      const roundedSum = Math.round(sum * 10) / 10; // snap to 0.1
      if (roundedSum !== 1.0) {
        return failSearch('Modality weights must sum exactly to 1.');
      }
    }

    // Build params for hook (safe after all validations)
    const params = {
      top: k,
      pooling_method: 'max',
      models: modelsPayload,
    };
    if (n >= 2) params.model_weights = selectedModels.map(k => Number(weights[k]));

    if (activeKeys.length >= 2) {
      const picked = {};
      ['text', 'image', 'transcription', 'metadata'].forEach((key) => {
        if (resolvedModes[key]) picked[key] = Number((mw ?? {})[key] ?? 0);
      });
      params.modality_weights = picked;
    }

    if (resolvedModes.image && imageFile) 
      params.image_query = imageFile;
    if (resolvedModes.transcription && transcriptQuery?.trim()) 
      params.transcription_query = transcriptQuery.trim();
    if (resolvedModes.metadata && metadataQuery?.trim()) 
      params.metadata_query = metadataQuery.trim();
    if (temporalMode && results?.length) 
      params.previous_results = results;

    // Call the hook
    const { ok, error: err } = await search(q, params);
    
    // On text search success -> push to history
    if (ok && resolvedModes.text && q) {
      pushHistory(q);
    }

    // Optional: If server still returns error, mirror behavior
    if (!ok && err) {
      clearResults();
    }

    // Turn off temporal mode after search
    if (temporalMode) setTemporalMode(false);
  };

  const onRewrite = async () => {
    setError('');
    const q = (query || '').trim();
    if (!q) return failSearch('Please enter a text query.');

    const first = selectedModels[0];
    const clip_model = first ? MODEL_MAP[first] : undefined;

    const { ok, data } = await rewrite({
      text: (query || '').trim(),
      model: 'gemini-2.5-flash-lite',
      clip_model,
      thinking: false,
    });

    if (ok && data?.rewritten_query) {
      setQuery(data.rewritten_query);
    }
  };

  return (
    <Home
      // Query and configuration
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
      searchHistory={searchHistory}

      // Results and interactions
      results={results}
      selectedItem={selectedItem}
      setSelectedItem={setSelectedItem}
      temporalMode={temporalMode}
      setTemporalMode={setTemporalMode}

      // API actions
      onSearch={onSearch}
      onRewrite={onRewrite}
      onSimilarSearch={(file) =>
        onSearch({
          modes: { text: false, image: true, transcription: false, metadata: false },
          imageFile: file,
        })
      }
      isSearching={isSearching}
      isRewriting={isRewriting}
      error={error}
    />
  );
}