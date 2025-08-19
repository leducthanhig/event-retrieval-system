import { useRef, useMemo, useEffect } from 'react';

export default function ImagePicker({ file, setFile }) {
  const inputRef = useRef(null);

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
        borderRadius: 12,
        padding: 8
      }}
    >
      {url && (
        <img
          src={url}
          alt="query"
          style={{ maxWidth: '100%', borderRadius: 8, display: 'block' }}
        />
      )}
      <button
        type="button"
        onClick={clear}
        aria-label="Remove image"
        title="Remove image"
        style={{
          position: 'absolute',
          top: 8,
          right: 8,
          border: '1px solid #4b5563',
          background: '#111827',
          color: '#f9fafb',
          borderRadius: 9999,
          width: 28,
          height: 28,
          cursor: 'pointer'
        }}
      >
        ×
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
        border: '2px dashed #6b7280',
        borderRadius: 12,
        padding: 16,
        minHeight: 140,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        color: '#e5e7eb',
        cursor: 'pointer',
        userSelect: 'none'
      }}
      title="Click, paste, or drag & drop an image"
    >
      <span style={{ fontSize: 22, lineHeight: 1 }}>＋</span>
      <span>Add image (click / paste / drop)</span>
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
