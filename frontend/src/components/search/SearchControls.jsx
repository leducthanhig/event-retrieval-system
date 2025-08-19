import { useEffect, useRef, useState } from 'react';

export default function SearchControls({ query, setQuery, onSearch, onRewrite, loading, error, compact, isSearching, isRewriting }) {
  const taRef = useRef(null);
  const [reachedMax, setReachedMax] = useState(false);
  const [maxPx, setMaxPx] = useState(() => Math.floor(window.innerHeight * 0.4));

  // Auto-resize textarea to fit content height
  const autoResize = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';

    // Limit max height to avoid covering the whole sidebar
    const maxPx = Math.floor(window.innerHeight * 0.4); // ~40% viewport
    const next = Math.min(ta.scrollHeight, maxPx);
    ta.style.height = `${next}px`;
    setReachedMax(ta.scrollHeight > maxPx);
  };

  useEffect(() => {
    requestAnimationFrame(autoResize);
  }, [query, maxPx]);

  useEffect(() => {
    const onResize = () => setMaxPx(Math.floor(window.innerHeight * 0.4));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const handleChange = (e) => {
    setQuery(e.target.value);
    requestAnimationFrame(autoResize);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSearch();
    }
  };

  const disabled = loading;

  return (
    <div>
      {/* Textarea for query input */}
      <textarea
        className="query-textarea"
        ref={taRef}
        type="text"
        placeholder="Enter your text query"
        value={query}
        onChange={handleChange}
        onInput={autoResize} 
        onKeyDown={handleKeyDown}
        spellCheck={false}
        rows={2}
        style={{ 
          width: '100%', 
          boxSizing: 'border-box',
          padding: '8px', 
          borderRadius: compact ? 4 : 6,
          border: '1px solid #d1d5db',
          marginBottom: 8,
          resize: 'none',                                // auto-resize, user won't drag the corner
          overflowY: reachedMax ? 'auto' : 'hidden',     // show scrollbar only if it hits max height
          maxHeight: `${maxPx}px`,
          lineHeight: 1.4,
          fontFamily: 'inherit',
          fontSize: 14, 
        }}
        aria-label="Query"
      />
      
      {/* Buttons for search and rewrite */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className="ner-btn"
          onClick={onSearch}
          disabled={isSearching}
        >
          {isSearching ? 'Searching' : 'Search'}
        </button>

        <button
          className="ner-btn"
          onClick={onRewrite}
          disabled={isRewriting}
        >
          {isRewriting ? 'Rewriting' : 'Rewrite'}
        </button>
      </div>
      
      {/* Error message*/}
      {error ? (
        <div style={{ color: '#b91c1c', marginTop: 8, fontSize: 14 }}>{error}</div>
      ) : null}
    </div>
  );
}