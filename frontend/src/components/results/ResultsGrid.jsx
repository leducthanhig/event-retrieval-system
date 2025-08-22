import ResultCard from './ResultCard';

export default function ResultsGrid({ results, onSelect, onSimilarSearch, error }) {
  if ((!results || results.length === 0) && !error) {
    return (
      <div style={{ padding: '1rem', color: '#9ca3af', textAlign: 'left' }}>
        No results found.
      </div>
    );
  }

  return (
    <div className="results-grid">
      {results.map((item, idx) => (
        <ResultCard 
          key={idx} 
          item={item} 
          onSelect={() => onSelect(item)}
          onSimilarSearch={onSimilarSearch}  
        />
      ))}
    </div>
  );
}