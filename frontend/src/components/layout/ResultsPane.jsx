import { useMemo, useState } from 'react';
import '../../styles/results.css';
import ResultsGrid from '../results/ResultsGrid';
import VideoPreviewModal from '../results/VideoPreview';

export default function ResultsPane({ results, onSelect, selectedItem, onClosePreview, onSimilarSearch, error }) {
  
  const [groupByVideo, setGroupByVideo] = useState(false);
  const [groupSortKey, setGroupSortKey] = useState('score');

  const getVideoId = (it) => it.video_id ?? it.videoId ?? it.video?.id ?? 'unknown';
  const getVideoTitle = (it) => it.video_title ?? it.videoTitle ?? it.video?.name ?? getVideoId(it);
  const getScore = (it) => Number(it.score ?? it.similarity ?? it.rank ?? 0);
  
  const parseShotId = (x) => {
    if (x == null) return Number.MAX_SAFE_INTEGER;
    if (typeof x === 'number') return x;
    const s = String(x);
    const m = s.replace(/^s/i, '').match(/\d+/);
    return m ? parseInt(m[0], 10) : Number.MAX_SAFE_INTEGER;
  };

  // Group & sort per video when enabled
  const grouped = useMemo(() => {
    if (!groupByVideo || !Array.isArray(results)) return null;
    const map = new Map();
    for (const it of results) {
      const vid = getVideoId(it);
      if (!map.has(vid)) map.set(vid, { title: getVideoTitle(it), items: [] });
      map.get(vid).items.push(it);
    }
    for (const [, group] of map) {
      group.items.sort((a, b) => {
        if (groupSortKey === 'shot') {
          const sa = parseShotId(a.shot_id ?? a.shotId ?? a.shot);
          const sb = parseShotId(b.shot_id ?? b.shotId ?? b.shot);
          if (sa !== sb) return sa - sb;
        }
        // default: score desc
        return getScore(b) - getScore(a);
      });
    }
    return Array.from(map, ([videoId, payload]) => ({
      videoId, title: payload.title, items: payload.items
    }));
  }, [groupByVideo, groupSortKey, results]);

  return (
    <main className="ner-results">
      {/* Top bar */}  
      <div className="results-topbar">
        <label className="topbar-item">
          <input
            type="checkbox"
            checked={groupByVideo}
            onChange={(e) => setGroupByVideo(e.target.checked)}
          />
          <span>Group by video</span>
        </label>

        {groupByVideo && (
          <label className="topbar-item">
            <span>Sort by:</span>
            <select
              value={groupSortKey}
              onChange={(e) => setGroupSortKey(e.target.value)}
            >
              <option value="score">Score (desc)</option>
              <option value="shot">ShotID (asc)</option>
            </select>
          </label>
        )}
      </div>

      <div className="results-scroll">
        {error ? (
          <div className="error-box">{error}</div>
        ) : null}
        
        {/* Flat mode */}
        {!groupByVideo && (
          <ResultsGrid
            results={results}
            onSelect={onSelect}
            onSimilarSearch={onSimilarSearch}
            error={error}
          />
        )}

        {/* Grouped mode */}
        {groupByVideo && (
          grouped && grouped.length > 0 ? (
            <div className="grouped-results">
              {grouped.map((g) => (
                <section key={g.videoId} className="video-group">
                  <div className="video-group__title">{g.title}</div>
                  <ResultsGrid
                    results={g.items}
                    onSelect={onSelect}
                    onSimilarSearch={onSimilarSearch}
                    disableInternalSort
                  />
                </section>
              ))}
            </div>
          ) : (
            <ResultsGrid
              results={[]}
              error={error}
              onSelect={onSelect}
              onSimilarSearch={onSimilarSearch}
              disableInternalSort
            />
          )
        )}

        <VideoPreviewModal 
          item={selectedItem} 
          onClose={onClosePreview}
        />
      </div>
    </main>
  );
}