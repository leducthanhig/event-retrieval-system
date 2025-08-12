import { useState } from 'react';
import SearchBar from './components/SearchBar';
import ModelSelector from './components/ModelSelector';
import SearchResult from './components/SearchResult';
import VideoPlayer from './components/VideoPlayer';
import './App.css';

const AVAILABLE_MODELS = [
  { label: 'ViT-L-16-SigLIP-256', model_name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
  { label: 'ViT-L-14-quickgelu', model_name: 'ViT-L-14-quickgelu', pretrained: 'dfn2b' }
];

function toAPISearchModel(uiModel) {
  return {
    name: uiModel.model_name,
    pretrained: uiModel.pretrained,
  };
}

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [selected, setSelected] = useState(null);

  const [mode, setMode] = useState('single'); // 'single' or 'multi'
  const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0]);
  const [weights, setWeights] = useState(['0.5', '0.5']); // as string

  const [isSearching, setIsSearching] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);

  const handleSearch = async () => {
    setError('');
    setHasSearched(true);

    // Ignore empty query
    if (!query.trim()) {
      setError('Please enter a query.');
      return;
    }

    // Validate input
    if (mode === 'single' && !selectedModel) {
      setError('Please select a model.');
      return;
    }
    if (mode === 'multi') {
      const w1 = parseFloat(weights[0]);
      const w2 = parseFloat(weights[1]);
      if (isNaN(w1) || isNaN(w2)) {
        setError('Please enter valid weights.');
        return;
      }
    }

    setIsSearching(true);

    try {
      let body = { pooling_method: 'max' };

      if (mode === 'single') {
        if (!selectedModel) {
          setError("Please select a model.");
          return;
        }
        body.models = [toAPISearchModel(selectedModel)];
      } else {
        body.models = [
          { name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
          { name: 'ViT-L-14-quickgelu', pretrained: 'dfn2b' }
        ];
        body.weights = weights.map((w) => parseFloat(w));
      }

      const response = await fetch(`http://localhost:8000/search?q=${encodeURIComponent(query)}&top=100`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error(`Failed to fetch (${response.status})`);
      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error('Search error:', err);
      setError('Error calling API: ' + err.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleRewrite = async () => {
    if (!query.trim()) return;
    setIsRewriting(true);

    const payload = {
      text: query,
      clip_model: {
        name: selectedModel.model_name,
        pretrained: selectedModel.pretrained,
      },
      thinking: true,
    };

    try {
      const response = await fetch('http://localhost:8000/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (data.rewritten_query) {
        setQuery(data.rewritten_query);
      } else {
        alert('Rewrite failed: No rewritten query returned.');
      }
    } catch (error) {
      console.error('Rewrite error:', error);
      alert('Rewrite failed due to network/server error.');
    } finally {
      setIsRewriting(false);
    }
  };

  return (
    <div style={{ padding: '1rem', fontFamily: 'sans-serif', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '2rem' }}>News Event Search</h1>

      {/* Search Input */}
      <SearchBar
        query={query}
        onChange={setQuery}
        onSubmit={handleSearch}
        onRewrite={handleRewrite}
        isSearching={isSearching}
        isRewriting={isRewriting}
      />

      {/* Model selection */}
      <div style={{ maxWidth: '500px', margin: '0 auto', padding: '0 1rem' }}>
        <ModelSelector
          mode={mode}
          setMode={setMode}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          weights={weights}
          setWeights={setWeights}
          AVAILABLE_MODELS={AVAILABLE_MODELS}
        />
      </div>

      {/* Error message */}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Results */}
      {hasSearched && (
        <div style={{ marginTop: '0.5rem' }}>
          <h2>Search Results</h2>
          <SearchResult isSearching={isSearching} results={results} onSelect={setSelected} />
        </div>
      )}

      {/* Video player */}
      {selected && (
        <VideoPlayer
          videoID={selected.video_id}
          shotID={selected.shot_id}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

export default App;
