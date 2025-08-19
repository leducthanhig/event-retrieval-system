import Sidebar from '../components/layout/Sidebar';
import ResultsPane from '../components/layout/ResultsPane';

export default function Home(props) {
  const {
    query, setQuery,
    selectedModels, setSelectedModels,
    weights, setWeights,
    results, setSelectedItem,
    onSearch, onRewrite, onUnifiedSearch, 
    loading, error,
    isSearching, isRewriting,
    selectedItem,
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
          onUnifiedSearch={onUnifiedSearch}
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
        />
      </main>
    </div>
  );
}