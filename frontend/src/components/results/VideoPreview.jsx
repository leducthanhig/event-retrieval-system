import { useRef, useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faArrowRight } from '@fortawesome/free-solid-svg-icons';

export default function VideoPreview({ data, onClose }) {
  if (!data) return null;

  const videoUrl = `/videos?url=${data.video_path}`;
  const videoRef = useRef(null);
  const frameTime = 1 / data.fps;
  const [currentFrame, setCurrentFrame] = useState(0); // Track the current frame index

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  const getVideoId = (it) => {
    if (!it) return 'unknown';
    if (it.video_id) return it.video_id;
    if (it.videoId) return it.videoId;
    if (it.video && it.video.id) return it.video.id;
    if (typeof it.video_path === 'string') {
      const parts = it.video_path.split(/[\\/]/);
      const base = parts[parts.length - 1] || '';
      const noExt = base.replace(/\.[^/.]+$/, '');
      return noExt || 'unknown';
    }
    return 'unknown';
  };

  const videoId = getVideoId(data);

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

  useEffect(() => {
    // Keep the frame display in sync while the video is playing/scrubbing
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
      e.preventDefault();
      stepFrames(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      stepFrames(1);
    }
  };

  // Step the video by a given number of frames (negative for backward, positive for forward)
  const stepFrames = (delta) => {
    const vid = videoRef.current;
    if (!vid || !Number.isFinite(vid.duration)) return;

    // Pause to ensure frame-accurate stepping
    if (!vid.paused) vid.pause();

    // Compute next frame index from current time -> index -> +delta
    const idxNow = Math.floor(vid.currentTime / frameTime);
    const idxNext = clamp(idxNow + delta, 0, Math.floor(vid.duration / frameTime));

    const newTime = idxNext * frameTime;
    vid.currentTime = clamp(newTime, 0, vid.duration);
    setCurrentFrame(idxNext);
    vid.focus();
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
          <strong>
            {videoId ? `${videoId}` : ''}
          </strong>
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
            <div style={{ marginTop: 0, color: '#fff' }}>
              <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                <label style={{ marginRight: 8 }}>Frame:</label>
                <input
                  type="number"
                  value={currentFrame}
                  onChange={(e) => setCurrentFrame(Number(e.target.value))}
                  style={{
                    width: 100,
                    height: 30,
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
                    border: 'none',
                    marginRight: 8
                  }}
                >
                  Go
                </button>

                <button
                  onClick={() => stepFrames(-1)}
                  className="icon-btn"
                  aria-label="Step backward one frame"
                  title="Previous frame"
                >
                  <FontAwesomeIcon icon={faArrowLeft} />
                </button>
                <button
                  onClick={() => stepFrames(1)}
                  className="icon-btn"
                  aria-label="Step forward one frame"
                  title="Next frame"
                >
                  <FontAwesomeIcon icon={faArrowRight} />
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
