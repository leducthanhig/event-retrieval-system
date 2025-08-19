import { useState } from 'react';
import SearchControls from '../search/SearchControls';
import ModelSelector from '../search/ModelSelector';
import SearchModeTabs from '../search/SearchModeTabs';
import ImagePicker from '../search/ImagePicker';

export default function Sidebar(props) {
  const {
    query, setQuery,
    selectedModels, setSelectedModels,
    weights, setWeights,
    onUnifiedSearch,
    onRewrite,
    loading, error,
    isSearching, isRewriting,
  } = props;

  // State for active tabs and image file
  const [activeTabs, setActiveTabs] = useState(['text']);
  const [imageFile, setImageFile] = useState(null);
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
        {/*System title + Tabs) */}
        <div style={{ 
          padding: '16px 12px 8px 12px', 
          //borderBottom: '1px solid #374151',
          position: 'sticky',
          top: 0,
          background: '#1e1e1e',
          zIndex: 2,
        }}>
          <div style={{ textAlign: 'center', wordBreak: 'break-word' }}>
            <div style={{ fontWeight: 700, lineHeight: 1.2, fontSize: 18, letterSpacing: 0.5 }}>
              EVENT RETRIEVAL SYSTEM
            </div>
            <div style={{ fontWeight: 500, marginTop: 4, fontSize: 16 }}>
              DoubleDT - AIC25
            </div>
          </div>

          {/* Tabs (TEXT / IMAGE / METADATA) */}
          <div style={{ marginTop: 16 }}>
            <SearchModeTabs activeTabs={activeTabs} setActiveTabs={setActiveTabs} />
          </div>
        </div>

        {/* Scrollable content */}
        <div style={{ flex: 1, padding: '12px' }}>
          {/* Text search */}
          {activeTabs.includes('text') && (
            <>
              <SearchControls
                query={query}
                setQuery={setQuery}
                onSearch={onUnifiedSearch}
                onRewrite={onRewrite}
                loading={loading}
                isSearching={isSearching}
                isRewriting={isRewriting}
                error={error}
                compact
              />
              <div style={{ marginTop: 16 }}>
                <ModelSelector
                  selectedModels={selectedModels}
                  setSelectedModels={setSelectedModels}
                  weights={weights}
                  setWeights={setWeights}
                />
              </div>
              <div style={{ borderTop: '1px solid #374151', margin: '16px 0' }} />
            </>
          )}

          {/* Image search */}
          {activeTabs.includes('image') && (
            <>
              <div style={{ marginTop: 16 }}>
                <ImagePicker file={imageFile} setFile={setImageFile} />
              </div>
              <div style={{ borderTop: '1px solid #374151', margin: '16px 0' }} />
            </>
          )}

          {/* Metadata search */}
          {activeTabs.includes('metadata') && (
            <div style={{ marginTop: 16 }}>
              <input
                type="text"
                placeholder="Enter metadata keywords..."
                value={metadataQuery}
                onChange={(e) => setMetadataQuery(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                }}
              />
            </div>
          )}
        </div>

        {/* Footer (sticky Search button) */}
        <div
          style={{
            borderTop: '1px solid #374151',
            padding: '12px',
            marginTop: 'auto',
            position: 'sticky',
            bottom: 0,
            background: '#1e1e1e',
            zIndex: 2
          }}
        >
          <button
            onClick={() => 
              onUnifiedSearch({ 
                modes: {
                  text: activeTabs.includes('text'),
                  image: activeTabs.includes('image'),
                  metadata: activeTabs.includes('metadata'),
               },
               imageFile,
               metadataQuery,
              })}
            disabled={loading || isSearching}
            style={{
              width: '100%',
              padding: '10px',
              fontSize: 15,
              fontWeight: 600,
              borderRadius: 6,
              border: '1px solid #d1d5db',
              backgroundColor: '#1a1a1a',
              color: 'white',
              cursor: loading || isSearching ? 'not-allowed' : 'pointer',
            }}
          >
            {isSearching ? 'Searching' : 'Search'}
          </button>
        </div>
      </div>
    </aside>
  );
}
