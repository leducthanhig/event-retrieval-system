import { useState } from 'react';

// Encapsulate /search (GET with query params) and /rewrite (POST).
export default function useSearch({ onResults }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Helper: build URL with safe query params
  const buildUrl = (base, params) => {
    const usp = new URLSearchParams();
    Object.entries(params || {}).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      // arrays -> repeat key
      if (Array.isArray(v)) {
        v.forEach((item) => usp.append(k, String(item)));
      } else {
        usp.set(k, String(v));
      }
    });
    return `${base}?${usp.toString()}`;
  };

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
    } catch (_e) {
      try { return await res.text(); } catch { return ''; }
    }
  };

  /**
   * Search: now GET /search with query params to match backend
   * Expected minimal params: q (text), top (default 100)
   * We also pass optional params to preserve features: pooling_method, mode, weights[], model_name, pretrained
   */
  const search = async (q, params) => {
    try {
      setLoading(true);
      setError('');

      const url = `http://localhost:8000/search?q=${encodeURIComponent(q)}&top=${encodeURIComponent(params?.top ?? 100)}`;

      // Body: using FormData (multipart/form-data)
      const fd = new FormData();
      fd.append('pooling_method', params?.pooling_method ?? 'max');
      if (params?.models) {
        fd.append('models', JSON.stringify(params.models));
      }
      if (Array.isArray(params?.clip_weights)) {
        fd.append('clip_weights', JSON.stringify(params.clip_weights.map(Number)));
      }
      if (Array.isArray(params?.weights)) {
        fd.append('weights', JSON.stringify(params.weights.map(Number)));
      }
      if (params?.image_query instanceof File || params?.image_query instanceof Blob) {
        fd.append('image_query', params.image_query);
      }

      const res = await fetch(url, {
        method: 'POST',
        body: fd,
      });

      if (!res.ok) {
        const msg = await extractError(res);
        setError(msg || `Search failed (${res.status})`);
        onResults?.([]);
        return;
      }

      const data = await res.json();
      onResults?.(Array.isArray(data?.results) ? data.results : data);
    } catch (e) {
      setError('Search failed due to network/server error.');
      onResults?.([]);
    } finally {
      setLoading(false);
    }
  };

  // /rewrite -> POST JSON to /rewrite
  const rewrite = async ({ text, model = 'gemini-2.5-flash-lite', clip_model, thinking = false }) => {
    try {
      setLoading(true);
      setError('');

      const payload = {
        text,
        model,
        thinking,
        ...(clip_model ? { clip_model } : {}),
      };

      const res = await fetch('http://localhost:8000/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const msg = await extractError(res);
        setError(msg || `Rewrite failed (${res.status})`);
        return;
      }
      
      const data = await res.json();
      return data;
    } catch (e) {
      setError('Rewrite failed due to network/server error.');
    } finally {
      setLoading(false);
    }
  };

  return { search, rewrite, loading, error, setError };
}
