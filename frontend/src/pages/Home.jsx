import Sidebar from '../components/layout/Sidebar';
import ResultsPane from '../components/layout/ResultsPane';

export default function Home(props) {
  const {
    query, setQuery,
    selectedModels, setSelectedModels,
    weights, setWeights,
    results, setSelectedItem,
    onSearch, onRewrite, 
    loading, error,
    isSearching, isRewriting,
    selectedItem,
    onSimilarSearch,
  } = props;

  return (
    <div className="ner-root">
      <aside className="ner-sidebar">
        <Sidebar
          query={query}
          setQuery={setQuery}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          weights={weights}
          setWeights={setWeights}
          onSearch={onSearch}
          onRewrite={onRewrite}
          loading={loading}
          isSearching={isSearching}
          isRewriting={isRewriting}
          error={error}
        />
      </aside>

      <main className="ner-results">
        <ResultsPane
          results={results}
          onSelect={setSelectedItem}
          selectedItem={selectedItem}
          onClosePreview={() => setSelectedItem(null)}
          onSimilarSearch={onSimilarSearch}
        />
      </main>
    </div>
  );
}