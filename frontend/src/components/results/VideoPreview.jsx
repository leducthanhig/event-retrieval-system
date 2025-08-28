import { useRef, useEffect, useState } from 'react';

export default function VideoPreview({ data, onClose }) {
  if (!data) return null;

  const videoUrl = `http://localhost:8000/${data.video_path}`;
  const videoRef = useRef(null);
  const frameTime = 1 / data.fps;
  const [currentFrame, setCurrentFrame] = useState(0); // Track the current frame index

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
        setCurrentFrame(Math.floor(data.start / frameTime));
      }
    };

    videoRef.current.addEventListener('loadedmetadata', handler);

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', handler);
      }
    };
  }, [data, frameTime]);

  useEffect(() => {
    const updateCurrentFrame = () => {
      if (videoRef.current) {
        setCurrentFrame(Math.floor(videoRef.current.currentTime / frameTime));
      }
    };

    if (videoRef.current) {
      videoRef.current.addEventListener('timeupdate', updateCurrentFrame);
    }

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('timeupdate', updateCurrentFrame);
      }
    };
  }, [frameTime]);

  const handleKeyDown = (e) => {
    if (!videoRef.current || !videoRef.current.paused) return;

    if (e.key === 'ArrowLeft') {
      e.preventDefault(); // Prevent default behavior
      // Seek one frame back
      const newTime = Math.max(0, videoRef.current.currentTime - frameTime);
      videoRef.current.currentTime = newTime;
      setCurrentFrame(Math.floor(newTime / frameTime));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault(); // Prevent default behavior
      // Seek one frame forward
      const newTime = Math.min(videoRef.current.duration, videoRef.current.currentTime + frameTime);
      videoRef.current.currentTime = newTime;
      setCurrentFrame(Math.floor(newTime / frameTime));
    }
  };

  const goToFrame = () => {
    if (videoRef.current) {
      const newTime = currentFrame * frameTime;
      videoRef.current.currentTime = Math.min(videoRef.current.duration, Math.max(0, newTime));
      setCurrentFrame(currentFrame);
      videoRef.current.focus(); // Refocus the video element
    }
  };

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
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              style={{ width: '100%', borderRadius: 6 }}
              onKeyDownCapture={handleKeyDown} // Add the keydown event listener here
              tabIndex="0" // Ensure the video element is focusable
            />
            <div style={{ marginTop: 8, color: '#fff' }}>
              <div style={{ display: 'flex', alignItems: 'center', marginTop: 8 }}>
                <label style={{ marginRight: 8 }}>Frame:</label>
                <input
                  type="number"
                  value={currentFrame}
                  onChange={(e) => setCurrentFrame(Number(e.target.value))}
                  style={{
                    marginRight: 8,
                    padding: '4px',
                    borderRadius: '4px',
                    border: '1px solid #ccc'
                  }}
                />
                <button
                  onClick={goToFrame}
                  style={{
                    padding: '4px 8px',
                    borderRadius: '4px',
                    background: '#4caf50',
                    color: '#fff',
                    border: 'none'
                  }}
                >
                  Go
                </button>
              </div>
            </div>
          </>
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
