import ResultCard from './ResultCard';

export default function ResultsGrid({ results, onSelect }) {
  if (!results || results.length === 0) {
    return (
      <div style={{ padding: '1rem', color: '#9ca3af', textAlign: 'center' }}>
        No results found.
      </div>
    );
  }

  return (
    <div className="results-grid">
      {results.map((item, idx) => (
        <ResultCard key={idx} item={item} onSelect={() => onSelect(item)} />
      ))}
    </div>
  );
}