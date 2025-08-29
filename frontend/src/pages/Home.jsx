import Sidebar from '../components/layout/Sidebar';
import ResultsPane from '../components/layout/ResultsPane';

export default function Home(props) {
  const {
    query, setQuery,
    topK, setTopK,
    activeTabs, setActiveTabs,
    modalityWeights, setModalityWeights,
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
      <Sidebar
        query={query}
        setQuery={setQuery}
        topK={topK}
        setTopK={setTopK}
        activeTabs={activeTabs}
        setActiveTabs={setActiveTabs}
        modalityWeights={modalityWeights}
        setModalityWeights={setModalityWeights}
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

      <ResultsPane
        results={results}
        onSelect={setSelectedItem}
        selectedItem={selectedItem}
        onClosePreview={() => setSelectedItem(null)}
        onSimilarSearch={onSimilarSearch}
        error={error}
      />
    </div>
  );
}
