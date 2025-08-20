import { useMemo } from 'react';

export default function ModelSelector({
  selectedModels,
  setSelectedModels,
  weights,
  setWeights,
  onValidationHint,
}) {
  // Model metadata used by the backend
  const MODEL_META = useMemo(
    () => ({
      siglip2:   { label: 'SIGLIP2',   name: 'ViT-L-16-SigLIP-384', pretrained: 'webli' },
      siglip:    { label: 'SIGLIP',    name: 'ViT-L-16-SigLIP-256', pretrained: 'webli' },
      quickgelu: { label: 'QUICKGELU', name: 'ViT-L-14-quickgelu',  pretrained: 'dfn2b'  },
    }),
    []
  );

  const options = ['siglip2', 'siglip', 'quickgelu'];
  const selectedCount = selectedModels.length;

  const handleWeightChange = (key, raw) => {
    if (raw === '') {
      setWeights(prev => ({ ...prev, [key]: '' }));
      return;
    }
    const cleaned = raw
      .replace(',', '.')
      .replace(/[^\d.]/g, '');
    if ((cleaned.match(/\./g) || []).length > 1) return;

    const num = Number(cleaned);
    if (Number.isNaN(num)) return;

    const clamped = Math.max(0, Math.min(1, num));
    const rounded1 = Math.round(clamped * 10) / 10;
    setWeights(prev => ({ ...prev, [key]: rounded1.toFixed(1) }));
  };

  // Toggle checkbox
  const toggleModel = (key) => {
    if (selectedModels.includes(key)) {
      const next = selectedModels.filter(k => k !== key);
      setSelectedModels(next);
    } else {
      setSelectedModels([...selectedModels, key]);
    }
  };

  // Compute sum for selected models (only when >= 2 selected)
  const sum = useMemo(() => {
    if (selectedCount < 2) return null;
    return selectedModels.reduce((acc, key) => acc + (Number(weights[key]) || 0), 0);
  }, [selectedModels, selectedCount, weights]);

  // Emit validation hint upward (optional)
  if (onValidationHint) {
    if (selectedCount >= 2) {
      const ok = Math.abs((sum ?? 0) - 1) < 1e-6;
      onValidationHint(ok ? '' : 'The sum of weights must equal 1.00.');
    } else {
      onValidationHint('');
    }
  } 

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', rowGap: 4 }}>
        {options.map((key) => {
          const checked = selectedModels.includes(key);
          const disabledWeight = selectedCount <= 1 || !checked;
          const shownValue = disabledWeight ? '' : (weights[key] ?? '');
          return (
            <div key={key} style={{ display: 'grid', gridTemplateColumns: '1fr 72px', gap: 8, alignItems: 'center' }}>
              {/* checkbox + label */}
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#e5e7eb', fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleModel(key)}
                />
                <span style={{ textTransform: 'none' }}>{MODEL_META[key].label}</span>
              </label>

              {/* weight input */}
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                disabled={disabledWeight}
                value={shownValue}
                onChange={(e) => handleWeightChange(key, e.target.value)}
                placeholder={disabledWeight ? '' : '0.0'}
                lang="en-US"
                style={{
                  width: '90%',
                  padding: '1px 6px',
                  borderRadius: 2,
                  border: '1px solid #d1d5db',
                  fontSize: 13,
                  background: disabledWeight ? '#2a2a2a' : 'white',
                  color: disabledWeight ? '#9ca3af' : '#111827',
                  boxSizing: 'border-box',
                  caretColor: 'transparent',
                }}
                inputMode="decimal"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
