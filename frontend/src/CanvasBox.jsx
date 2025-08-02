import { useRef, useState } from 'react';

function CanvasBox({ onBoxDrawn, objectLabels }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [start, setStart] = useState(null);
  const [current, setCurrent] = useState(null);
  const [finalBoxes, setFinalBoxes] = useState([]);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
  const [draggingIndex, setDraggingIndex] = useState(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // === Object name input modal ===
  const [showInput, setShowInput] = useState(false);
  const [inputPosition, setInputPosition] = useState({ x: 0, y: 0 });
  const [pendingBox, setPendingBox] = useState(null);
  const [inputValue, setInputValue] = useState('');

  const handleCanvasMouseDown = (e) => {
    if (e.button !== 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStart({ x, y });
    setCurrent(null);
    setDrawing(true);
    setSelectedBoxIndex(null);
  };

  const handleCanvasMouseMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (drawing) {
      setCurrent({ x: mouseX, y: mouseY });
    } else if (draggingIndex !== null) {
      const box = finalBoxes[draggingIndex];
      const newX = Math.max(0, Math.min(mouseX - dragOffset.x, canvasRef.current.offsetWidth - box.width));
      const newY = Math.max(0, Math.min(mouseY - dragOffset.y, canvasRef.current.offsetHeight - box.height));
      const updated = [...finalBoxes];
      updated[draggingIndex] = { ...box, x: Math.round(newX), y: Math.round(newY) };
      setFinalBoxes(updated);
    }
  };

  const handleCanvasMouseUp = () => {
    if (draggingIndex !== null) {
      setDraggingIndex(null);
      return;
    }

    if (start && current) {
      const x = Math.min(start.x, current.x);
      const y = Math.min(start.y, current.y);
      const width = Math.abs(current.x - start.x);
      const height = Math.abs(current.y - start.y);

      const canvas = canvasRef.current;
      const canvasWidth = canvas.offsetWidth;
      const canvasHeight = canvas.offsetHeight;

      const clampedX = Math.max(0, Math.min(x, canvasWidth));
      const clampedY = Math.max(0, Math.min(y, canvasHeight));
      const clampedWidth = Math.min(width, canvasWidth - clampedX);
      const clampedHeight = Math.min(height, canvasHeight - clampedY);

      const box = {
        x: Math.round(clampedX),
        y: Math.round(clampedY),
        width: Math.round(clampedWidth),
        height: Math.round(clampedHeight),
      };

      setPendingBox(box);
      setInputValue('');
      setInputPosition({ x: clampedX, y: clampedY });
      setShowInput(true);
    }

    setDrawing(false);
    setStart(null);
    setCurrent(null);
  };

  const handleBoxClick = (idx, e) => {
    e.stopPropagation();
    setSelectedBoxIndex(prev => (prev === idx ? null : idx));
  };

  const handleBoxMouseDown = (idx, e) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    const rect = canvasRef.current.getBoundingClientRect();
    const offsetX = e.clientX - rect.left - finalBoxes[idx].x;
    const offsetY = e.clientY - rect.top - finalBoxes[idx].y;
    setDraggingIndex(idx);
    setDragOffset({ x: offsetX, y: offsetY });
  };

  const handleRename = () => {
    const oldName = finalBoxes[selectedBoxIndex].objectName;
    setPendingBox(finalBoxes[selectedBoxIndex]);
    setInputValue(oldName);
    setInputPosition({ x: finalBoxes[selectedBoxIndex].x, y: finalBoxes[selectedBoxIndex].y });
    setShowInput(true);
  };

  const handleDelete = () => {
    const updated = finalBoxes.filter((_, idx) => idx !== selectedBoxIndex);
    setFinalBoxes(updated);
    setSelectedBoxIndex(null);
  };

  const handleInputSubmit = () => {
    const name = inputValue.trim();
    if (!name) return;
    const newBox = { ...pendingBox, objectName: name };
    let updated;

    // Update or add box
    const exists = finalBoxes.findIndex(b => b === pendingBox);
    if (exists !== -1) {
      updated = [...finalBoxes];
      updated[exists] = newBox;
    } else {
      updated = [...finalBoxes, newBox];
      onBoxDrawn(newBox);
    }

    setFinalBoxes(updated);
    setShowInput(false);
    setPendingBox(null);
    setInputValue('');
  };

  const handleCancelInput = () => {
    setShowInput(false);
    setPendingBox(null);
    setInputValue('');
  };

  return (
    <div
      ref={canvasRef}
      onMouseDown={handleCanvasMouseDown}
      onMouseMove={handleCanvasMouseMove}
      onMouseUp={handleCanvasMouseUp}
      onMouseLeave={() => setDraggingIndex(null)}
      onContextMenu={(e) => e.preventDefault()}
      style={{
        width: '640px',
        height: '360px',
        border: '2px dashed gray',
        position: 'relative',
        userSelect: 'none',
      }}
    >
      {finalBoxes.length === 0 && !drawing && !showInput && (
        <p style={{ textAlign: 'center', marginTop: '140px', color: 'gray' }}>
          Drag to label an object
        </p>
      )}


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

      {pendingBox && (
        <div
          style={{
            position: 'absolute',
            left: pendingBox.x,
            top: pendingBox.y,
            width: pendingBox.width,
            height: pendingBox.height,
            border: '2px dashed green',
            backgroundColor: 'rgba(0, 255, 0, 0.1)',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        />
      )}

      {finalBoxes.map((box, idx) => (
        <div
          key={idx}
          onMouseDown={(e) => handleBoxMouseDown(idx, e)}
          onClick={(e) => handleBoxClick(idx, e)}
          style={{
            position: 'absolute',
            left: box.x,
            top: box.y,
            width: box.width,
            height: box.height,
            border: idx === selectedBoxIndex ? '2px solid blue' : '2px solid green',
            backgroundColor: 'rgba(0, 255, 0, 0.1)',
            cursor: 'move',
            zIndex: idx === selectedBoxIndex ? 2 : 1,
          }}
        >
          <span style={{
            position: 'absolute',
            top: '-1.2em',
            left: 0,
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            color: 'black',
            padding: '1px 4px',
            fontSize: '0.75rem',
            borderRadius: '2px',
          }}>
            {box.objectName}
          </span>

          {selectedBoxIndex === idx && (
            <div style={{
              position: 'absolute',
              top: box.height + 4,
              left: 0,
              backgroundColor: '#eee',
              border: '1px solid #ccc',
              borderRadius: '4px',
              padding: '4px',
              zIndex: 3,
              display: 'flex',
              gap: '4px',
            }}>
              <button onClick={handleRename} style={{ fontSize: '0.75rem', padding: '2px 6px' }}>Rename</button>
              <button onClick={handleDelete} style={{ fontSize: '0.75rem', padding: '2px 6px' }}>Delete</button>
            </div>
          )}
        </div>
      ))}

      {/* === Custom input for object label === */}
      {showInput && (
        <div style={{
          position: 'absolute',
          left: inputPosition.x,
          top: inputPosition.y - 36,
          padding: '0px',
          zIndex: 1000,
          display: 'flex',
          gap: '4px',
        }}>
          <input
            list="object-list"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleInputSubmit();
              if (e.key === 'Escape') handleCancelInput();
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
            autoFocus
            placeholder="Enter object name..."
            style={{
              padding: '4px 6px',
              fontSize: '0.9rem',
              border: '1px solid #aaa',
              borderRadius: '4px',
              outline: 'none',
              width: '160px',
              background: 'white',
              color: 'black'
            }}
          />
          <button
            onClick={handleCancelInput}
            style={{
              fontSize: '0.8rem',
              padding: '4px 6px',
              border: '1px solid #aaa',
              borderRadius: '4px',
              background: '#f8f8f8',
              cursor: 'pointer',
              color: 'black'
            }}
          >
            Cancel
          </button>
          <datalist id="object-list">
            {inputValue.length >= 2 && objectLabels.map((label, idx) => (
              <option key={idx} value={label} />
            ))}
          </datalist>
        </div>
      )}
    </div>
  );
}

export default CanvasBox;
