import { useState } from 'react';
import SearchModeTabs from '../search/SearchModeTabs';
import TextSearch from '../search/TextSearch';
import ModelSelector from '../search/ModelSelector';
import ImageSearch from '../search/ImageSearch';
import MetaSearch from '../search/MetaSearch';
import TranscriptSearch from '../search/TranscriptSearch';

export default function Sidebar(props) {
  const {
    query, setQuery,
    topK, setTopK,
    selectedModels, setSelectedModels,
    weights, setWeights,
    onSearch,
    onRewrite,
    loading, error,
    isSearching, isRewriting,
  } = props;

  const [activeTabs, setActiveTabs] = useState(['text']);
  const [imageFile, setImageFile] = useState(null);
  const [transcriptQuery, setTranscriptQuery] = useState('');
  const [metadataQuery, setMetadataQuery] = useState('');

  return (
    <aside className="ner-sidebar sidebar-scroll">
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          maxWidth: '100%',
          width: '100%',
          padding: 0,
        }}
      >
        {/* Header */}
        <div style={{ 
          padding: '16px 12px 8px 12px',
          position: 'sticky',
          top: 0,
          background: '#1e1e1e',
          zIndex: 2,
        }}>
          {/* System's name */}
          <div style={{ textAlign: 'center', wordBreak: 'break-word' }}>
            <div style={{ fontWeight: 700, lineHeight: 1.2, fontSize: 18, letterSpacing: 0.5 }}>
              EVENT RETRIEVAL SYSTEM
            </div>
            <div style={{ fontWeight: 500, marginTop: 4, fontSize: 16 }}>
              DoubleDT - AIC25
            </div>
          </div>

          {/* Search Mode Tabs */}
          <div style={{ marginTop: 16 }}>
            <SearchModeTabs activeTabs={activeTabs} setActiveTabs={setActiveTabs} />
          </div>
        </div>

        {/* Searching content */}
        <div style={{ flex: 1, padding: '12px' }}>
          {/* Text search */}
          {activeTabs.includes('text') && (
            <>
              <TextSearch
                query={query}
                setQuery={setQuery}
                onSearch={onSearch}
                loading={loading}
                error={error}
                compact
              />
              <div style={{ marginTop: 4 }}>
                <ModelSelector
                  selectedModels={selectedModels}
                  setSelectedModels={setSelectedModels}
                  weights={weights}
                  setWeights={setWeights}
                />
              </div>
              {(activeTabs.includes('image') || activeTabs.includes('transcript') || activeTabs.includes('metadata')) && (
                <div style={{ borderTop: '1px solid #374151', margin: '16px 0' }} />
              )}
            </>
          )}

          {/* Image search */}
          {activeTabs.includes('image') && (
            <>
              <div style={{ marginTop: 16 }}>
                <ImageSearch file={imageFile} setFile={setImageFile} />
              </div>
              {(activeTabs.includes('metadata') || activeTabs.includes('transcript')) && (
                <div style={{ borderTop: '1px solid #374151', margin: '16px 0' }} />
              )}
            </>
          )}

          {/* Transcript search */}
          {activeTabs.includes('transcript') && (
            <>
              <div style={{ marginTop: 16 }}>
                <TranscriptSearch
                  value={transcriptQuery}
                  setValue={setTranscriptQuery}
                  error={error}
                />
              </div>
              {activeTabs.includes('metadata') && (
                <div style={{ borderTop: '1px solid #374151', margin: '16px 0' }} />
              )}
            </>
          )}

          {/* Metadata search */}
          {activeTabs.includes('metadata') && (
            <div style={{ marginTop: 16 }}>
              <MetaSearch
                value={metadataQuery}
                setValue={setMetadataQuery}
                error={error}
              />
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            borderTop: '1px solid #374151',
            padding: '8px 10px',
            marginTop: 'auto',
            position: 'sticky',
            bottom: 0,
            background: '#1e1e1e',
            zIndex: 2,
            display: 'flex',
            gap: 8,
          }}
        >
          <button
              onClick={() => 
                onSearch({ 
                  modes: {
                    text: activeTabs.includes('text'),
                    image: activeTabs.includes('image'),
                    transcript: activeTabs.includes('transcript'),
                    metadata: activeTabs.includes('metadata'),
                },
                imageFile,
                transcriptQuery,
                metadataQuery,
                })}
              disabled={loading || isSearching}
              style={{
                width: '100%',
                padding: '8px 10px',
                fontSize: 16,
                fontWeight: 600,
                borderRadius: 18,
                backgroundColor: '#532d8d',
                color: 'white',
                cursor: loading || isSearching ? 'not-allowed' : 'pointer',
                flex: 3,
              }}
            >
              {isSearching ? 'Searching' : 'Search'}
            </button>

            <button
              onClick={onRewrite}
              disabled={loading || isRewriting || !activeTabs.includes('text')}
              style={{ 
                width: '100%',
                padding: '8px 10px',
                fontSize: 16,
                fontWeight: 600,
                borderRadius: 18,
                //border: '1px solid #d1d5db',
                backgroundColor: '#1a1a1a',
                color: 'white',
                cursor: loading || isSearching ? 'not-allowed' : 'pointer',
                flex: 1,
              }}
            >
              {isRewriting ? 'Rewriting' : 'Rew'}
            </button>

            {/* Input top-k*/}
            <input
              type="number"
              min={1}
              max={500}
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              title="Top-k (1–500)"
              style={{
                width: 65,
                padding: '8px 8px',
                fontSize: 14,
                borderRadius: 4,
                border: '0.7px solid #d1d5db',
                backgroundColor: '#111827',
                color: '#f9fafb',
                boxSizing: 'border-box',
                outline: 'none'
              }}
            />
        </div>
      </div>
    </aside>
  );
}
