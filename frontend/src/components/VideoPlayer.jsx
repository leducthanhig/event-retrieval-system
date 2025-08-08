import React, { useEffect, useRef, useState } from 'react';

const VideoPlayer = ({ videoID, shotID, onClose }) => {
  const [videoUrl, setVideoUrl] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    const fetchVideoData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/shots/${videoID}/${shotID}`);
        if (!response.ok) throw new Error('Failed to fetch video info');
        const data = await response.json();

        setVideoUrl(`http://localhost:8000/${data.video_path}`);
        setStartTime(data.start);
      } catch (err) {
        console.error('Error loading video:', err);
      }
    };

    fetchVideoData();
  }, [videoID, shotID]);

  useEffect(() => {
    const onLoaded = () => {
      if (videoRef.current && startTime !== null) {
        videoRef.current.currentTime = startTime;
        videoRef.current.play();
      }
    };

    const video = videoRef.current;
    if (video) {
      video.addEventListener('loadedmetadata', onLoaded);
    }

    return () => {
      if (video) {
        video.removeEventListener('loadedmetadata', onLoaded);
      }
    };
  }, [videoUrl, startTime]);

  return (
    <div
      style={{
        position: 'fixed',
        top: 0, left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 999
      }}
      onClick={onClose} // Click outside closes the player
    >
      <div
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside video
        style={{
          background: '#fff',
          borderRadius: '8px',
          padding: '1px',
          border: '1px solid #ccc',
          maxWidth: '800px',
          width: '90%',
          position: 'relative'
        }}
      >
        {videoUrl ? (
          <video
            ref={videoRef}
            controls
            style={{ width: '100%', borderRadius: '6px' }}
            src={videoUrl}
          />
        ) : (
          <p style={{ textAlign: 'center' }}>Loading video...</p>
        )}
      </div>
    </div>
  );
};

export default VideoPlayer;
