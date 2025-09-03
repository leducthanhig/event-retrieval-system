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

  const onSearch = async ({
    modes,
    imageFile,
    transcriptQuery,
    metadataQuery,
    modalityWeights: mwOverride,
  } = {}) => {
    setError('');
    const resolvedModes = modes ?? { text: true, image: false, transcription: false, metadata: false };

    // Map selected models to backend payload format
    const n = selectedModels.length;
    const modelsPayload =
      n === 1 ? MODEL_MAP[selectedModels[0]] : selectedModels.map((k) => MODEL_MAP[k]);

    // Parse topK with safe fallback
    const top = parseInt(topK, 10) || 100;

    // Prepare params for the hook
    const params = {
      top,
      pooling_method: 'max',
      models: modelsPayload,
    };

    // If multiple models: attach model_weights
    if (n >= 2) {
      params.model_weights = selectedModels.map((k) => Number(weights[k]));
    }

    // Handle modality weights when more than one mode is active
    const mw = mwOverride ?? modalityWeights ?? {};
    if (Object.values(resolvedModes).filter(Boolean).length >= 2) {
      const picked = {};
      ['text', 'image', 'transcription', 'metadata'].forEach((k) => {
        if (resolvedModes[k]) picked[k] = Number(mw[k] ?? 0);
      });
      params.modality_weights = picked;
    }

    // Add optional queries for each modality
    const q = (query || '').trim();
    if (resolvedModes.image && imageFile) {
      params.image_query = imageFile;
    }
    if (resolvedModes.transcription && transcriptQuery?.trim()) {
      params.transcription_query = transcriptQuery.trim();
    }
    if (resolvedModes.metadata && metadataQuery?.trim()) {
      params.metadata_query = metadataQuery.trim();
    }
    if (temporalMode && results?.length) {
      params.previous_results = results;
    }

    await search(q, params);

    // Turn off temporal mode after search
    if (temporalMode) setTemporalMode(false);
  };

  const onRewrite = async () => {
    setError('');

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