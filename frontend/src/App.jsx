import { useState } from 'react';
import SearchBar from './components/SearchBar';
import ModelSelector from './components/ModelSelector';
import SearchResult from './components/SearchResult';
import VideoPlayer from './components/VideoPlayer';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [selected, setSelected] = useState(null);

  const [mode, setMode] = useState('single'); // 'single' or 'multi'
  const [selectedModel, setSelectedModel] = useState({
    model_name: 'ViT-L-16-SigLIP-256',
    pretrained: 'webli'
  });
  const [weights, setWeights] = useState(['0.5', '0.5']); // as string

  const handleSearch = async () => {
    setError('');
    setHasSearched(true);

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

    try {
      let body = { pooling_method: 'max' };

      if (mode === 'single') {
        if (!selectedModel) {
          setError("Please select a model.");
          return;
        }
        body.models = [selectedModel];
      } else {
        body.models = [
          { model_name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
          { model_name: 'ViT-L-14-quickgelu', pretrained: 'dfn2b' }
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
        />
      </div>

      {/* Error message */}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Results */}
      {hasSearched && (
        <div style={{ marginTop: '0.5rem' }}>
          <h2>Search Results</h2>
          <SearchResult results={results} onSelect={setSelected} />
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
