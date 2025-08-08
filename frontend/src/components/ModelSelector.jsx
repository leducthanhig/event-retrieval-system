const AVAILABLE_MODELS = [
  { label: 'ViT-L-16-SigLIP-256', model_name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
  { label: 'ViT-L-14-quickgelu', model_name: 'ViT-L-14-quickgelu', pretrained: 'dfn2b' }
];

function ModelSelector({ mode, setMode, selectedModel, setSelectedModel, weights, setWeights }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', gap: '2rem' }}>
      {/* Single model */}
      <div style={{ flex: 1 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <input
            type="radio"
            value="single"
            checked={mode === 'single'}
            onChange={() => setMode('single')}
            style={{ display: 'none' }}
          />
          <div
            style={{
              width: '16px',
              height: '16px',
              border: '1px solid #999',
              borderRadius: '2px',
              backgroundColor: mode === 'single' ? '#2196f3' : 'white',
              color: 'white',
              fontSize: '12px',
              textAlign: 'center',
              lineHeight: '16px',
              fontWeight: 'bold'
            }}
          >
            {mode === 'single' ? '✓' : ''}
          </div>
          Use 1 Model
        </label>

        <div style={{ paddingLeft: '0rem' }}>
          {AVAILABLE_MODELS.map((model) => (
            <div key={model.model_name} style={{ display: 'flex', alignItems: 'center', marginTop: '0.5rem', marginBottom: '0.3rem' }}>
              <input
                type="radio"
                name="model"
                value={model.model_name}
                checked={selectedModel?.model_name === model.model_name}
                onChange={() => setSelectedModel(model)}
                disabled={mode !== 'single'}
                style={{ marginRight: '0.5rem' }}
              />
              <span>{model.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Multi model */}
      <div style={{ flex: 1 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <input
            type="radio"
            value="multi"
            checked={mode === 'multi'}
            onChange={() => setMode('multi')}
            style={{ display: 'none' }}
          />
          <div
            style={{
              width: '16px',
              height: '16px',
              border: '1px solid #999',
              borderRadius: '2px',
              backgroundColor: mode === 'multi' ? '#2196f3' : 'white',
              color: 'white',
              fontSize: '12px',
              textAlign: 'center',
              lineHeight: '16px',
              fontWeight: 'bold'
            }}
          >
            {mode === 'multi' ? '✓' : ''}
          </div>
          Use 2 Models
        </label>

        <div style={{ paddingLeft: '0rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', marginTop: '0.5rem', marginBottom: '0.5rem' }}>
            <span style={{ width: '80px' }}>Weight 1:</span>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={weights[0]}
               onChange={(e) => {
                const raw = parseFloat(e.target.value);
                if (isNaN(raw)) return;
                const clamped = Math.min(1, Math.max(0, raw));
                const rounded = Math.round(clamped * 10) / 10;
                const other = Math.round((1 - rounded) * 10) / 10;
                setWeights([rounded, other]);
              }}
              disabled={mode !== 'multi'}
              style={{ width: '60px', marginLeft: '0.5rem' }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center' }}>
            <span style={{ width: '80px' }}>Weight 2:</span>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={weights[1]}
              readOnly
              disabled={mode !== 'multi'}
              style={{ width: '60px', marginLeft: '0.5rem' }}
            />
          </label>
        </div>
      </div>
    </div>
  );
}

export default ModelSelector;
