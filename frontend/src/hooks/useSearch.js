import { useRef, useState } from 'react';

export default function useSearch({ onResults } = {}) {
  const [isSearching, setIsSearching] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);
  const [error, setError] = useState('');

  // Keep a reference to the active search request to support cancelation
  const activeSearchController = useRef(null);

  /** Helper: join base + path safely */
  const joinUrl = (path) => path;

  /** Helper: extract a meaningful error string out of a Response */
  const extractError = async (res) => {
    try {
      const data = await res.json();
      if (data?.detail) {
        if (Array.isArray(data.detail)) {
          return data.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
        }
        if (typeof data.detail === 'string') return data.detail;
      }
      return JSON.stringify(data);
    } catch {
      try {
        return await res.text();
      } catch {
        return '';
      }
    }
  };

  const search = async (q, params = {}) => {
    // Cancel previous search if still running
    if (activeSearchController.current) {
      activeSearchController.current.abort();
    }
    const controller = new AbortController();
    activeSearchController.current = controller;

    try {
      setIsSearching(true);
      setError('');

      // Build URL with top as query param (server reads top from query)
      const top = params?.top ?? 100;
      const url = joinUrl(`/search?top=${encodeURIComponent(top)}`);

      // Build multipart body
      const fd = new FormData();

      // Text query
      if (typeof q === 'string' && q.trim().length > 0) {
        fd.append('text_query', q.trim());
      }

      // Core knobs
      fd.append('pooling_method', params?.pooling_method ?? 'max');

      // Model selection / weights
      if (params?.models) {
        fd.append('models', JSON.stringify(params.models));
      }
      if (Array.isArray(params?.model_weights)) {
        fd.append('model_weights', JSON.stringify(params.model_weights.map(Number)));
      }
      if (Array.isArray(params?.clip_weights)) {
        fd.append('clip_weights', JSON.stringify(params.clip_weights.map(Number)));
      }

      // Multimodal knobs
      if (params?.modality_weights && typeof params.modality_weights === 'object') {
        fd.append('modality_weights', JSON.stringify(params.modality_weights));
      }
      if (params?.image_query instanceof File || params?.image_query instanceof Blob) {
        fd.append('image_query', params.image_query);
      }
      if (typeof params?.transcription_query === 'string' && params.transcription_query.trim()) {
        fd.append('transcription_query', params.transcription_query.trim());
      }
      if (typeof params?.metadata_query === 'string' && params.metadata_query.trim()) {
        fd.append('metadata_query', params.metadata_query.trim());
      }

      // Temporal search
      if (params?.previous_results) {
        fd.append('previous_results', JSON.stringify(params.previous_results));
      }

      // Fire request
      const res = await fetch(url, {
        method: 'POST',
        body: fd,
        signal: controller.signal,
      });

      // Handle non-2xx
      if (!res.ok) {
        const msg = await extractError(res);
        setError(msg || `Search failed (${res.status})`);
        onResults?.([]);
        return { ok: false, error: msg || `HTTP ${res.status}` };
      }

      // Parse JSON
      const data = await res.json();
      const items = Array.isArray(data?.results) ? data.results : data;
      onResults?.(items);
      return { ok: true, data: items };
    } catch (err) {
      if (err?.name !== 'AbortError') {
        setError('Search failed due to network/server error.');
        onResults?.([]);
      }
      return { ok: false, error: err?.message };
    } finally {
      // Clear controller if this call is the active one
      if (activeSearchController.current === controller) {
        activeSearchController.current = null;
      }
      setIsSearching(false);
    }
  };

  const rewrite = async (payload) => {
    console.log('[onRewrite] called');
    try {
      setIsRewriting(true);
      setError('');

      const url = joinUrl('/rewrite');
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload ?? {}),
      });

      if (!res.ok) {
        const msg = await extractError(res);
        setError(msg || `Rewrite failed (${res.status})`);
        return { ok: false, error: msg || `HTTP ${res.status}` };
      }

      const data = await res.json();
      return { ok: true, data };
    } catch {
      setError('Rewrite failed due to network/server error.');
      return { ok: false, error: 'Network/Server error' };
    } finally {
      setIsRewriting(false);
    }
  };

  return {
    search,
    rewrite,
    isSearching, 
    isRewriting,
    error,
    setError,
  };
}
