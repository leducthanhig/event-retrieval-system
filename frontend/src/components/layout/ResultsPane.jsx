import ResultsGrid from '../results/ResultsGrid';
import VideoPreviewModal from '../results/VideoPreview';

export default function ResultsPane({ results, onSelect, selectedItem, onClosePreview, onSimilarSearch }) {
  return (
    <main className="ner-results">
        <div className="results-scroll">
            <ResultsGrid 
              results={results} 
              onSelect={onSelect}
              onSimilarSearch={onSimilarSearch}
            />
            <VideoPreviewModal 
              item={selectedItem} 
              onClose={onClosePreview}
            />
        </div>
    </main>
  );
}