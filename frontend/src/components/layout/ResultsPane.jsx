import ResultsGrid from '../results/ResultsGrid';
import VideoPreviewModal from '../results/VideoPreview';

export default function ResultsPane({ results, onSelect, selectedItem, onClosePreview, onSimilarSearch, error }) {
  return (
    <main className="ner-results">
        <div className="results-scroll">
          {error ? (
            <div
              style={{
                margin: '12px',
                padding: '8px 12px',
                border: '1px solid #7f1d1d',
                background: '#431313',
                color: '#fecaca',
                borderRadius: 8,
                fontSize: 14
              }}
            >
              {error}
            </div>
          ) : null}
          
          <ResultsGrid 
            results={results} 
            onSelect={onSelect}
            onSimilarSearch={onSimilarSearch}
            error={error}
          />
          <VideoPreviewModal 
            item={selectedItem} 
            onClose={onClosePreview}
          />
        </div>
    </main>
  );
}