import { useEffect, useState } from 'react';

export default function InfoModal({ open, onClose, item }) {
  const [copyMessage, setCopyMessage] = useState(''); // State to hold the copy message

  // Close on ESC
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const { score, video_id, shot_id, thumbnail } = item || {};

  const match = thumbnail?.match(/F(\d+)_selected\.jpg$/);
  const frameNumber = match ? Number(match[1]) : null;

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopyMessage('Copied to clipboard!'); // Set the copy message
      setTimeout(() => setCopyMessage(''), 1000); // Clear the message after 1 seconds
    }).catch((err) => {
      console.error('Failed to copy text: ', err);
    });
  };

  return (
    <div
      aria-modal="true"
      role="dialog"
      aria-labelledby="info-title"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.45)',
        zIndex: 1000,
        padding: 16,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(420px, 92vw)',
          background: '#1f2937',
          color: '#f9fafb',
          borderRadius: 6,
          border: '1px solid #4b5563',
          boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
          padding: 16,
        }}
      >
        <h2 id="info-title" style={{ marginTop: 0, marginBottom: 12, fontSize: 16 }}>
          DETAILS
        </h2>

        <div style={{ fontSize: 14, lineHeight: 1.6 }}>
          <div><strong>Score:</strong> {typeof score === 'number' ? score.toFixed(4) : score}</div>
          {video_id !== undefined && (
            <div onClick={() => handleCopy(video_id)} style={{ cursor: 'pointer' }}>
              <strong>Video ID:</strong> {video_id}
            </div>
          )}
          {shot_id !== undefined && <div><strong>Shot ID:</strong> {shot_id}</div>}
          {frameNumber && (
            <div
              onClick={() => handleCopy(frameNumber)}
              style={{ cursor: 'pointer' }}
            >
              <strong>Frame:</strong> {frameNumber}
            </div>
          )}
        </div>

        {/* Display the copy message */}
        {copyMessage && (
          <div style={{ marginTop: 12, fontSize: 12, color: '#10b981', textAlign: 'center' }}>
            {copyMessage}
          </div>
        )}

        <div style={{ marginTop: 12, fontSize: 12, color: '#9ca3af', textAlign: 'left' }}>
          Press 'Esc' or click on backdrop to close
        </div>
      </div>
    </div>
  );
}
