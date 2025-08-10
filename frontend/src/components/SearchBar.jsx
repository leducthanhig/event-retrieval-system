import { useRef, useEffect } from 'react';

function Spinner({ size = 1.2 }) {
  const pixelSize = `${size}em`;
  const viewBoxSize = 24;
  const center = viewBoxSize / 2;
  const radius = center - 2;

  return (
    <svg
      width={pixelSize}
      height={pixelSize}
      viewBox={`0 0 ${viewBoxSize} ${viewBoxSize}`}
      aria-hidden
      style={{ flexShrink: 0 }}
    >
      <circle
        cx={center}
        cy={center}
        r={radius}
        stroke="currentColor"
        strokeWidth="3"
        fill="none"
        opacity="0.25"
      />
      <path
        d={`M${viewBoxSize} ${center}a${radius} ${radius} 0 0 1-${radius} ${radius}`}
        stroke="currentColor"
        strokeWidth="3"
        fill="none"
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`0 ${center} ${center}`}
          to={`360 ${center} ${center}`}
          dur="0.8s"
          repeatCount="indefinite"
        />
      </path>
    </svg>
  );
}

function SearchBar({ query, onChange, onSubmit, onRewrite, isSearching, isRewriting }) {
  const textareaRef = useRef(null);

  // Auto-resize textarea height based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto'; // reset before measuring
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [query]);

  const buttonStyle = {
    padding: '0.5rem 1rem',
    width: '130px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    whiteSpace: 'nowrap',
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '1rem',
        marginBottom: '1.5rem',
        alignItems: 'flex-start',
      }}
    >
      <textarea
        ref={textareaRef}
        placeholder="Enter your query..."
        value={query}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // prevent newline
            onSubmit();
          }
        }}
        spellCheck={false}
        rows={1}
        style={{
          width: '400px',
          maxWidth: '480px',
          padding: '0.5rem',
          fontSize: '1rem',
          fontFamily: 'inherit',
          resize: 'none',
          lineHeight: '1.4',
          overflow: 'hidden',
          borderRadius: '4px',
          minHeight: '45px',
        }}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <button
          style={buttonStyle}
          onClick={onSubmit}
          disabled={isSearching}
          aria-busy={isSearching}
        >
          {isSearching && <Spinner />}
          {isSearching ? 'Searching' : 'Search'}
        </button>

        <button
          style={buttonStyle}
          onClick={onRewrite}
          disabled={isRewriting}
          aria-busy={isRewriting}
        >
          {isRewriting && <Spinner />}
          {isRewriting ? 'Rewriting' : 'Rewrite'}
        </button>
      </div>
    </div>
  );
}

export default SearchBar;
