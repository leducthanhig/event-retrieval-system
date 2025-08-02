import { useState, useEffect } from 'react';
import CanvasBox from './CanvasBox';
import VideoPlayer from './VideoPlayer';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [selected, setSelected] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [objectLabels, setObjectLabels] = useState([]);

  // Load object labels from JSON file
  useEffect(() => {
  fetch('/obj_name.json')
    .then(res => res.json())
    .then(data => setObjectLabels(data))
    .catch(err => console.error("Failed to load object labels:", err));
  }, []);

  // Callback when a new box is drawn on canvas
  const handleBoxDrawn = (box) => {
    setBoxes(prev => [...prev, box]);
  };

  // Handle search action
  const handleSearch = async () => {
    setError('');
    setHasSearched(true);

    try {
      // Prepare request body according to backend schema
      const body = {
        bboxes: boxes.map(b => ({
          label: b.objectName,
          xmin: b.x,
          ymin: b.y,
          xmax: b.x + b.width,
          ymax: b.y + b.height
        })),
        weights: [1.0, 0.0],
        pooling_method: 'max'
      };

      const response = await fetch(`http://localhost:8000/search?q=${encodeURIComponent(query)}&top=10`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body)
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
    <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
      <h1>News Event Search</h1>

      {/* Text query input and Search button */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem' }}>
        <input
          type="text"
          placeholder="Enter your query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch();
          }}
          style={{ width: '60%', maxWidth: '480px', padding: '0.5rem', fontSize: '1rem' }}
        />
        <button
          style={{ padding: '0.5rem 1rem' }}
          onClick={handleSearch}
        >
          Search
        </button>
      </div>

      {/* Display error message if any */}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Canvas for drawing bounding boxes */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}>
        <CanvasBox onBoxDrawn={handleBoxDrawn} objectLabels={objectLabels} />
      </div>

      {/* Search Results Section */}
      {hasSearched && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Search Results</h2>
          {results.length === 0 ? (
            <p>No results found.</p>
          ) : (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(6, 1fr)',
                gap: '0.5rem',
              }}
            >
              {results.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    height: '145px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    border: '1px solid #ccc',
                    padding: '0.1rem',
                    borderRadius: '8px',
                    textAlign: 'center',
                    position: 'relative',
                  }}
                >
                  <img
                    src={`http://localhost:8000/${item.thumbnail}`}
                    alt="Thumbnail"
                    style={{
                      width: '100%',
                      height: 'auto',
                      maxHeight: '120px',
                      cursor: 'pointer',
                      borderRadius: '4px',
                      objectFit: 'cover'
                    }}
                    onClick={() => setSelected(item)}
                  />
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'center',
                      marginTop: '0rem',
                      gap: '0.7rem',
                    }}
                  >
                    <button
                      onClick={() => setSelected(item)}
                      title="View Video"
                      style={{
                        fontSize: '0.9rem',
                        borderRadius: '4px',
                        background: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      📺
                    </button>
                    <button
                      onClick={() => alert(JSON.stringify(item, null, 2))}
                      title="Details"
                      style={{
                        fontSize: '0.9rem',
                        borderRadius: '4px',
                        background: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      ℹ️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Video Player Section */}
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
