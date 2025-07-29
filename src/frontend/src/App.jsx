import { useState } from 'react';
import CanvasBox from './CanvasBox';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [selected, setSelected] = useState(null);
  const [boxes, setBoxes] = useState([]);

  const handleBoxDrawn = (box) => {
    setBoxes([box]); // hoặc push thêm nếu cần lưu nhiều box
  };


  const handleSearch = async () => {
    setError('');
    setHasSearched(true);
    try {
      const response = await fetch(`http://localhost:8000/search?q=${encodeURIComponent(query)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "obj-loc": boxes
        })
      });


      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setError('Error calling API');
    }
  };

  return (
    <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
      <h1>News Event Search</h1>

      {/* Text query input */}
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

      {/* Debug: Hiển thị bounding box đã vẽ */}
      {boxes.length > 0 && (
        <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#555' }}>
          Bounding box: {boxes[0].objectName} at ({boxes[0].x}, {boxes[0].y}) size {boxes[0].width}x{boxes[0].height}
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}>
        <CanvasBox onBoxDrawn={handleBoxDrawn} />
      </div>

      {/* Search Results area */}
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
                gap: '1rem',
              }}
            >
              {results.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    border: '1px solid #ccc',
                    padding: '0.5rem',
                    borderRadius: '8px',
                    textAlign: 'center',
                    position: 'relative',
                  }}
                >
                  <img
                    src={`http://localhost:8000/${item.thumbnail}`} // http://localhost:8000
                    alt="Thumbnail"
                    style={{
                      width: '100%',
                      cursor: 'pointer',
                      borderRadius: '4px',
                    }}
                    onClick={() => setSelected(item)}
                  />
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'center',
                      marginTop: '0.5rem',
                      gap: '1rem',
                    }}
                  >
                    <button onClick={() => setSelected(item)} title="View Video">📺</button>
                    <button onClick={() => alert(JSON.stringify(item, null, 2))} title="Details">ℹ️</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Video player for selected frame */}
      {selected && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Video Preview</h2>
          <p>
            <strong>Video ID:</strong> {selected.videoId}, <strong>Start:</strong> {selected.start}s, <strong>End:</strong> {selected.end}s
          </p>
          <video width="640" height="360" controls autoPlay>
            <source
              src={`http://localhost:8000/video?id=${selected.videoId}&start=${selected.start}&end=${selected.end}`}
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
        </div>
      )}
    </div>
  );
}

export default App;
