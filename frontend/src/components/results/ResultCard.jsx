import { useState } from 'react';
import InfoModal from './InfoModal';
import VideoPreview from './VideoPreview';

// Render one thumbnail + action area below (info + preview buttons)
export default function ResultCard({ item, onSelect, onSimilarSearch }) {
  const [openInfo, setOpenInfo] = useState(false);
  const [previewData, setPreviewData] = useState(null);

  const thumbUrl = item?.thumbnail ? `/${item.thumbnail}` : '';

  const handlePreview = async () => {
    try {
      const resp = await fetch(`/shots/${item.video_id}/${item.shot_id}`);
      const data = await resp.json();
      setPreviewData(data);
    } catch (err) {
      console.error('Failed to fetch shot data', err);
    }
  };

  const handleSimilarSearch = async () => {
    try {
      if (!thumbUrl) return;
      const resp = await fetch(thumbUrl, { mode: 'cors' });
      if (!resp.ok) throw new Error(`fetch thumbnail failed: ${resp.status}`);
      const blob = await resp.blob();

      const type = blob.type || 'image/jpeg';
      const ext = type.includes('png') ? 'png' : 'jpg';

      const file = new File([blob], `similar.${ext}`, { type });

      onSimilarSearch && onSimilarSearch(file);
    } catch (err) {
      console.error('Similar search failed', err);
    }
  };

  return (
    <div className="result-card">
      <img
        className="thumb"
        src={thumbUrl}
        alt="Thumbnail"
        onClick={onSelect}
      />

      {/* Actions area below thumbnail */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 8, paddingTop: 0 }}>
        <button
          className="action-btn"
          title="Show Info"
          onClick={() => setOpenInfo(true)}
        >
          ℹ️
        </button>

        <button
          className="action-btn"
          title="Preview Video"
          onClick={handlePreview}
        >
          ▶️
        </button>

        <button
          className="action-btn"
          title="Similar Search"
          onClick={handleSimilarSearch}
        >
          🔍
        </button>
      </div>

      <InfoModal
        open={openInfo}
        onClose={() => setOpenInfo(false)}
        item={item}
      />

      {previewData && (
        <VideoPreview
          data={previewData}
          onClose={() => setPreviewData(null)}
        />
      )}
    </div>
  );
}