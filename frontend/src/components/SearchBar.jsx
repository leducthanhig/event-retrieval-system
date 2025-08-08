function SearchBar({ query, onChange, onSubmit }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
      <input
        type="text"
        placeholder="Enter your query..."
        value={query}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') onSubmit();
        }}
        style={{ width: '400px', maxWidth: '480px', padding: '0.5rem', fontSize: '1rem' }}
      />
      <button style={{ padding: '0.5rem 1rem' }} onClick={onSubmit}>
        Search
      </button>
    </div>
  );
}

export default SearchBar;
