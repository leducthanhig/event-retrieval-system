import { useRef, useState, useEffect } from 'react';

export default function MetaSearch({ value, setValue, error }) {
  const taRef = useRef(null);
  const [reachedMax, setReachedMax] = useState(false);
  const [maxPx, setMaxPx] = useState(() => Math.floor(window.innerHeight * 0.4));

  // Auto-resize textarea
  const autoResize = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    const next = Math.min(ta.scrollHeight, maxPx);
    ta.style.height = `${next}px`;
    setReachedMax(ta.scrollHeight > maxPx);
  };

  useEffect(() => {
    requestAnimationFrame(autoResize);
  }, [value, maxPx]);

  useEffect(() => {
    const onResize = () => setMaxPx(Math.floor(window.innerHeight * 0.3));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const handleChange = (e) => {
    setValue(e.target.value);
    requestAnimationFrame(autoResize);
  };

  return (
    <div>
      <textarea
        className="query-textarea"
        ref={taRef}
        placeholder="Enter metadata keywords"
        value={value}
        onChange={handleChange}
        onInput={autoResize}
        spellCheck={false}
        rows={2}
        style={{
          width: '100%',
          boxSizing: 'border-box',
          padding: '8px',
          borderRadius: 4,
          border: '1px solid #d1d5db',
          resize: 'none',
          overflowY: reachedMax ? 'auto' : 'hidden',
          maxHeight: `${maxPx}px`,
          lineHeight: 1.4,
          fontFamily: 'inherit',
          fontSize: 14,
        }}
        aria-label="Metadata query"
      />
      {error ? (
        <div style={{ color: '#b91c1c', marginTop: 8, fontSize: 14 }}>{error}</div>
      ) : null}
    </div>
  );
}
