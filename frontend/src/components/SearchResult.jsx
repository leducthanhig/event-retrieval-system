import React from 'react';

function SearchResult({ isSearching, results, onSelect }) {
  if (isSearching) {
    return <p>Searching...</p>
  }

  if (results.length === 0) {
    return <p>No results found.</p>;
  }

  return (
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
              objectFit: 'cover',
            }}
            onClick={() => onSelect(item)}
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
              onClick={() => onSelect(item)}
              title="View Video"
              style={{
                fontSize: '0.9rem',
                borderRadius: '4px',
                background: 'none',
                cursor: 'pointer',
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
                cursor: 'pointer',
              }}
            >
              ℹ️
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default SearchResult;
