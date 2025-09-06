import { useRef, useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faArrowRight } from '@fortawesome/free-solid-svg-icons';

export default function VideoPreview({ data, onClose }) {
  if (!data) return null;

  const videoUrl = `/videos?url=${encodeURIComponent(data.video_path)}`;
  const videoRef = useRef(null);

  // Frame math helpers
  const frameTime = 1 / data.fps;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  const [currentFrame, setCurrentFrame] = useState('0');

  // Accuracy / retry constants for seeking
  const EPS_BASE = Math.max(1e-3, frameTime / 1000); // ~1ms (or smaller than frameTime)
  const MAX_RETRIES = 3;

  // Seeking state
  const seekingRef = useRef(false);
  const queuedDeltaRef = useRef(0);
  const targetFrameRef = useRef(0);

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

  // Close on Escape (window-level)
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Arrow keys (window-level)
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stepFrames(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stepFrames(1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Seek to data.start as soon as metadata is ready (with small epsilon)
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || data.start === undefined) return;

    const EPS = Math.max(1e-3, frameTime / 1000);
    const setStart = () => {
      const safeDur = Number.isFinite(vid.duration) ? vid.duration : data.start + EPS;
      const t = clamp(data.start + EPS, 0, safeDur);
      vid.currentTime = t;
      setCurrentFrame(Math.floor(t / frameTime));
      // make sure the video can receive focus if user wants to use space/play, etc.
      vid.focus?.();
    };

    if (vid.readyState >= 1) {
      setStart();
      return;
    }

    const onMeta = () => setStart();
    vid.addEventListener('loadedmetadata', onMeta, { once: true });
    return () => vid.removeEventListener('loadedmetadata', onMeta);
  }, [data.start, frameTime]);

  // Keep currentFrame in sync with the video time (prefer rVFC, fallback to timeupdate)
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;

    let handle = null;
    let running = true;

    const updateFromTime = () => {
      if (!running || !videoRef.current) return;
      setCurrentFrame(Math.floor(videoRef.current.currentTime / frameTime));
    };

    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const tick = () => {
        if (!running) return;
        updateFromTime();
        // @ts-ignore – rVFC is supported on modern browsers
        handle = vid.requestVideoFrameCallback(tick);
      };
      // @ts-ignore
      handle = vid.requestVideoFrameCallback(tick);
      return () => {
        running = false;
        if (vid && handle != null && 'cancelVideoFrameCallback' in HTMLVideoElement.prototype) {
          // @ts-ignore
          vid.cancelVideoFrameCallback(handle);
        }
      };
    }

    // Fallback for older browsers
    const onTime = updateFromTime;
    vid.addEventListener('timeupdate', onTime);
    return () => {
      vid.removeEventListener('timeupdate', onTime);
    };
  }, [frameTime]);

  // time <-> frame helpers with clamping
  const timeToFrame = (t) => Math.floor(t / frameTime);
  const clampFrame = (idx) => {
    const vid = videoRef.current;
    if (!vid || !Number.isFinite(vid.duration)) return Math.max(0, idx);
    const maxIdx = Math.floor(vid.duration / frameTime);
    return Math.min(Math.max(0, idx), maxIdx);
  };

  // Accurate seek with epsilon and small retries, also queues extra deltas while seeking
  const seekToFrameAccurate = (frameIdx, attempt = 0) => {
    const vid = videoRef.current;
    if (!vid || !Number.isFinite(vid.duration)) return;

    if (!vid.paused) vid.pause(); // ensure frame-accurate stepping

    seekingRef.current = true;
    targetFrameRef.current = frameIdx;

    const eps = EPS_BASE * (attempt + 1); // 0.001, 0.002, 0.003, ...
    const targetTime = clamp(frameIdx * frameTime + eps, 0, vid.duration);

    const onSeeked = () => {
      const gotIdx = timeToFrame(vid.currentTime);

      if (gotIdx === frameIdx && Math.abs(vid.currentTime - targetTime) < 0.02) {
        // Success
        setCurrentFrame(gotIdx);
        seekingRef.current = false;

        const q = queuedDeltaRef.current;
        queuedDeltaRef.current = 0;
        if (q !== 0) {
          const next = clampFrame(targetFrameRef.current + q);
          seekToFrameAccurate(next, 0);
        }
      } else if (attempt + 1 < MAX_RETRIES) {
        // Retry with a slightly larger epsilon
        seekToFrameAccurate(frameIdx, attempt + 1);
      } else {
        // Give up after MAX_RETRIES; still update UI to nearest frame
        setCurrentFrame(gotIdx);
        seekingRef.current = false;

        const q = queuedDeltaRef.current;
        queuedDeltaRef.current = 0;
        if (q !== 0) {
          const next = clampFrame(targetFrameRef.current + q);
          seekToFrameAccurate(next, 0);
        }
      }

      vid.removeEventListener('seeked', onSeeked);
    };

    vid.addEventListener('seeked', onSeeked, { once: true });
    vid.currentTime = targetTime;
  };

  // Public: step by delta frames (used by both keyboard and icon buttons)
  const stepFrames = (delta) => {
    const vid = videoRef.current;
    if (!vid || !Number.isFinite(vid.duration)) return;

    const idxNow = timeToFrame(vid.currentTime);
    const target = clampFrame(idxNow + delta);

    if (seekingRef.current) {
      // Queue delta while we are still seeking
      queuedDeltaRef.current = Math.max(-9999, Math.min(9999, queuedDeltaRef.current + delta));
      return;
    }

    seekToFrameAccurate(target, 0);
  };

  // Jump to frame (Go button or Enter in input)
  const goToFrame = () => {
    const vid = videoRef.current;
    if (!vid || !Number.isFinite(vid.duration)) return;

    const target = clampFrame(currentFrame);
    if (seekingRef.current) {
      // If user requests a jump while seeking, override the target
      targetFrameRef.current = target;
      queuedDeltaRef.current = 0;
      return;
    }
    seekToFrameAccurate(target, 0);
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
              preload="auto"
              style={{ width: '100%', maxHeight: '75vh', borderRadius: 6 }}
              tabIndex="0" // Ensure the video element is focusable
            />
            <div style={{ marginTop: 0, color: '#fff' }}>
              <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                <label style={{ marginRight: 8 }}>Frame:</label>
                <input
                  type="number"
                  value={currentFrame}
                  onChange={(e) => setCurrentFrame(e.target.value)} // keep raw string
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const num = Number(currentFrame);
                      if (!Number.isNaN(num)) {
                        setCurrentFrame(String(num)); // normalize on commit
                        goToFrame();
                      }
                    }
                  }}
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
