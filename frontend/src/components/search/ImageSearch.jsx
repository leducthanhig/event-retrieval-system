import { useState, useRef, useMemo, useEffect } from 'react';

export default function ImageSearch({ file, setFile }) {
  const inputRef = useRef(null);
  const [showDelete, setShowDelete] = useState(false);

  // Create object URL for the image file
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    if (!url) return;
    return () => URL.revokeObjectURL(url);
  }, [url]);

  // Open system file picker
  const openPicker = () => inputRef.current?.click();

  const onFiles = (files) => {
    const f = files?.[0];
    if (!f) return;
    if (!f.type.startsWith('image/')) return; // ignore non-image
    setFile(f);
  };

  // Drag & drop
  const onDrop = (e) => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };
  const onDragOver = (e) => e.preventDefault();

  // Paste
  const onPaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const it of items) {
      if (it.type.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) { setFile(f); break; }
      }
    }
  };

  // Remove image
  const clear = (e) => {
    e.stopPropagation();
    setFile(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return file ? (
    <div
      style={{
        position: 'relative',
        border: '1px solid #374151',
        borderRadius: 8,
        padding: 2
      }}
      onMouseEnter={() => setShowDelete(true)}
      onMouseLeave={() => setShowDelete(false)}
      onFocus={() => setShowDelete(true)}
      onBlur={() => setShowDelete(false)}
      tabIndex={0}
    >
      {url && (
        <img
          src={url}
          alt="query"
          style={{ maxWidth: '100%', borderRadius: 6.5, display: 'block' }}
        />
      )}
      <button
        type="button"
        onClick={clear}
        aria-label="Remove image"
        title="Remove image"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'absolute',
          top: 6,
          right: 6,
          background: 'rgba(17, 24, 39, 0.7)',
          color: '#f9fafb',
          width: 24,
          height: 24,
          borderRadius: '50%',
          fontSize: 10,
          lineHeight: 1,
          cursor: 'pointer',
          opacity: showDelete ? 1 : 0,
          pointerEvents: showDelete ? 'auto' : 'none',
          transition: 'opacity 120ms ease'
        }}
      >
        ✕
      </button>
    </div>
  ) : (
    <div
      onClick={openPicker}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onPaste={onPaste}
      tabIndex={0}
      style={{
        border: '1.5px dashed #6b7280',
        borderRadius: 8,
        padding: 16,
        height: 140,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#e5e7eb',
        cursor: 'pointer',
        userSelect: 'none'
      }}
      title="Click, paste, or drag & drop an image"
    >
      <span style={{ fontSize: 40, lineHeight: 'normal', opacity: 0.7 }}> + </span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => onFiles(e.target.files)}
        style={{ display: 'none' }}
      />
    </div>
  );
}
