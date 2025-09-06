import { useEffect, useRef, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faWandMagicSparkles, faClockRotateLeft } from '@fortawesome/free-solid-svg-icons';

export default function TextSearch({ query, setQuery, onSearch, onRewrite, isRewriting, loading, historyItems = [] }) {
  const taRef = useRef(null);
  const [reachedMax, setReachedMax] = useState(false);
  const [maxPx, setMaxPx] = useState(() => Math.floor(window.innerHeight * 0.4));
  const [showHistory, setShowHistory] = useState(false);

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
  const disableHistory = loading || isRewriting;

  // Pick a history item: replace the entire query
  const pickHistory = (txt) => {
    setQuery(txt);
    setShowHistory(false);
    requestAnimationFrame(() => taRef.current?.focus());
  };

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
        rows={3}
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

      {!isRewriting ? (
        <FontAwesomeIcon
          icon={faWandMagicSparkles}
          title='Rewrite query'
          className={`rewrite-icon-svg${disableRewrite ? ' disabled' : ''}`}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => !disableRewrite && onRewrite?.(query)}
          aria-hidden="true"
        />
      ) : (
        <div className="rewrite-icon-svg">
          <div className="rewrite-spinner" aria-hidden="true"></div>
        </div>
      )}

      <FontAwesomeIcon
        icon={faClockRotateLeft}
        title="Search history"
        className={`history-icon-svg${disableHistory ? ' disabled' : ''}${showHistory ? ' active' : ''}`}
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => !disableHistory && setShowHistory(v => !v)}
        aria-expanded={showHistory ? 'true' : 'false'}
        aria-controls="history-panel"
        aria-pressed={showHistory ? 'true' : 'false'}
      />

      {/* History dropdown/panel */}
      {showHistory && historyItems?.length > 0 && (
        <div
          id="history-panel"
          className="history-panel"
          role="listbox"
          aria-label="Text search history"
        >
          {historyItems.map((it, idx) => (
            <button
              key={idx}
              type="button"
              role="option"
              className="history-item"
              onClick={() => pickHistory(it.text)}
              title={it.text}
            >
              {it.text}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}