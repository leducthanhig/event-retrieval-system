import { useRef, useEffect } from 'react';

export default function VideoPreview({ data, onClose }) {
  if (!data) return null;

  const videoUrl = `http://localhost:8000/${data.video_path}`;
  const videoRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!videoRef.current || data.start === undefined) return;

    const handler = () => {
      if (videoRef.current) {
        videoRef.current.currentTime = data.start;
      }
    };

    videoRef.current.addEventListener('loadedmetadata', handler);

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', handler);
      }
    };
  }, [data]);

  return (
    <div style={backdropStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <strong>PREVIEW</strong>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>
            Press 'Esc' or click on backdrop to close
          </div>
        </div>

        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay
            style={{ width: '100%', borderRadius: 6 }}
          />
        ) : (
          <div>No video URL found for this result.</div>
        )}
      </div>
    </div>
  );
}

const backdropStyle = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
};

const modalStyle = {
  width: 'min(900px, 90vw)', background: '#1f2937', borderRadius: 6,
  padding: 6, boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
};