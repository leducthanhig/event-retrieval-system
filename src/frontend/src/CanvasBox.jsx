import { useRef, useState } from 'react';

function CanvasBox({ onBoxDrawn }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [start, setStart] = useState(null);
  const [current, setCurrent] = useState(null);
  const [finalBox, setFinalBox] = useState(null);

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    setStart({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setCurrent(null);
    setDrawing(true);
  };

  const handleMouseMove = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    setCurrent({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleMouseUp = () => {
    if (start && current) {
      const x = Math.min(start.x, current.x);
      const y = Math.min(start.y, current.y);
      const width = Math.abs(current.x - start.x);
      const height = Math.abs(current.y - start.y);

      const objectName = prompt("Enter object name for this bounding box:", "person") || "unknown";

      const box = {
        x: Math.round(x),
        y: Math.round(y),
        width: Math.round(width),
        height: Math.round(height),
        objectName: objectName.trim(),
      };


      setFinalBox(box);           // Lưu box đã vẽ để hiển thị
      onBoxDrawn(box);            // Gửi box về App.jsx
    }

    setDrawing(false);
    setStart(null);
    setCurrent(null);
  };

  return (
    <div
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{
        width: '640px',
        height: '360px',
        border: '2px dashed gray',
        position: 'relative',
        userSelect: 'none',
      }}
    >
      {/* Canvas Area Label */}
      {!finalBox && !drawing && (
        <p style={{ textAlign: 'center', marginTop: '140px', color: 'gray' }}>
          Canvas Area (16:9)
        </p>
      )}

      {/* Đang vẽ box (theo chuột) */}
      {drawing && start && current && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(start.x, current.x),
            top: Math.min(start.y, current.y),
            width: Math.abs(current.x - start.x),
            height: Math.abs(current.y - start.y),
            border: '2px solid red',
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Box sau khi đã vẽ xong */}
      {finalBox && (
        <div
          style={{
            position: 'absolute',
            left: finalBox.x,
            top: finalBox.y,
            width: finalBox.width,
            height: finalBox.height,
            border: '2px solid green',
            backgroundColor: 'rgba(0, 255, 0, 0.1)',
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  );
}

export default CanvasBox;
