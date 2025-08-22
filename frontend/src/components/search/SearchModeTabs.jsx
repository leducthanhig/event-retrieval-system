import React from 'react';

const TABS = [
  { id: 'text', text: 'TEXT' },
  { id: 'image', text: 'IMAGE' },
  { id: 'transcription', text: 'AUDIO' },
  { id: 'metadata', text: 'FILTER' },
];

export default function SearchModeTabs({ activeTabs, setActiveTabs, modalityWeights, setModalityWeights }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 0,
        marginBottom: 4,
      }}
    >
      {TABS.map(tab => {
        const active = activeTabs.includes(tab.id);
        const value = modalityWeights[tab.id] ?? 0;

        const onChangeWeight = (e) => {
          let v = parseFloat(e.target.value);
          if (Number.isNaN(v)) v = 0;
          v = Math.max(0, Math.min(1, Math.round(v * 10) / 10));
          setModalityWeights(w => ({ ...w, [tab.id]: v }));
        };

        return (
          <div key={tab.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <button
              key={tab.id}
              onClick={() =>
                setActiveTabs(prev =>
                  prev.includes(tab.id) ? prev.filter(t => t !== tab.id) : [...prev, tab.id]
                )
              } 
              aria-pressed={active}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                borderRadius: 0,
                outline: 'none',
                cursor: 'pointer',
                padding: '4px 2px',
                textAlign: 'center',
                width: '100%',
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

            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={value}
              onChange={onChangeWeight}
              disabled={!active}
              title={`Weight for ${tab.text} (0.0–1.0)`}
              style={{
                alignItems: 'center',
                justifyContent: 'center',
                marginTop: 6,
                width: '80%', 
                padding: '4px 6px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid #4b5563',
                background: '#111827',
                color: '#f9fafb',
                opacity: active ? 1 : 0.3,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
