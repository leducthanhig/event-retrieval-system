import { useEffect, useRef, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faWandMagicSparkles } from '@fortawesome/free-solid-svg-icons';
import { faCircleNotch } from '@fortawesome/free-solid-svg-icons';

export default function TextSearch({ query, setQuery, onSearch, onRewrite, isRewriting, loading }) {
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

  const disableRewrite = loading || isRewriting || !query?.trim();

  return (
    <div className="query-field">
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
        readOnly={isRewriting}
        aria-busy={isRewriting ? 'true' : 'false'}
        aria-label="Query"
        style={{ 
          width: '100%', 
          boxSizing: 'border-box',
          padding: '8px 36px 8px 8px',
          borderRadius: 4,
          border: '1px solid #d1d5db',
          marginBottom: 8,
          resize: 'none',                                // auto-resize, user won't drag the corner
          overflowY: reachedMax ? 'auto' : 'hidden',     // show scrollbar only if it hits max height
          maxHeight: `${maxPx}px`,
          lineHeight: 1.4,
          fontFamily: 'inherit',
          fontSize: 14, 
        }}
      />

      <FontAwesomeIcon
        icon={faWandMagicSparkles}
        title='Rewrite query'
        className={`rewrite-icon-svg${disableRewrite ? ' disabled' : ''}`}
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => !disableRewrite && onRewrite?.(query)}
        aria-hidden="true"
      />

      {isRewriting && (
        <div className="query-overlay" aria-hidden="true">
          <FontAwesomeIcon icon={faCircleNotch} className="query-overlay-spinner" spin />
        </div>
      )}

    </div>
  );
}