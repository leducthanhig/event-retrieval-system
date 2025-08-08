import { useRef, useEffect } from 'react';

function SearchBar({ query, onChange, onSubmit }) {
  const textareaRef = useRef(null);

  // Auto-resize textarea height based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto'; // reset before measuring
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [query]);

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
        <button style={{ padding: '0.5rem 1rem' }} onClick={onSubmit}>
          Search
        </button>
        <button style={{ padding: '0.5rem 1rem' }}>
          Rewrite
        </button>
      </div>
    </div>
  );
}

export default SearchBar;
