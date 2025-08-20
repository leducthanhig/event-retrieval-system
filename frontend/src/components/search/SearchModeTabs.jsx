import React from 'react';

const TABS = [
  { id: 'text', text: 'TEXT' },
  { id: 'image', text: 'IMAGE' },
  { id: 'transcript', text: 'AUDIO' },
  { id: 'metadata', text: 'FILTER' },
];

export default function SearchModeTabs({ activeTabs, setActiveTabs }) {
  const toggleTab = (id) => {
    setActiveTabs(prev =>
      prev.includes(id) ? prev.filter(t => t !== id) : [...prev, id]
    );
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 0,
        marginBottom: 4
      }}
    >
      {TABS.map(tab => {
        const active = activeTabs.includes(tab.id);
        return (
          <button
            key={tab.id}
            onClick={() => toggleTab(tab.id)}
            aria-pressed={active}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              borderRadius: 0,
              outline: 'none',
              cursor: 'pointer',
              padding: '8px 2px',
              textAlign: 'center',
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: 0.5,
              color: active ? '#3b82f6' : '#e5e7eb',
              borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
              transition: 'color 120ms ease, border-color 120ms ease'
            }}
          >
            {tab.text}
          </button>
        );
      })}
    </div>
  );
}
