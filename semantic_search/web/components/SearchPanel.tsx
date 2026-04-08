"use client";

import { useState } from "react";
import type { SearchHit, SearchResponse } from "@/lib/types";

function apiBase(): string {
  const raw =
    process.env.NEXT_PUBLIC_SEARCH_API_URL ?? "http://127.0.0.1:8000";
  return raw.replace(/\/$/, "");
}

export default function SearchPanel() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchHit[]>([]);

  async function onSearch(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResults([]);
    const q = query.trim();
    if (!q) {
      setError("Enter a search query.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${apiBase()}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k }),
      });
      const text = await res.text();
      if (!res.ok) {
        let detail = text;
        try {
          const j = JSON.parse(text) as { detail?: string | unknown };
          if (typeof j.detail === "string") detail = j.detail;
        } catch {
          /* keep raw */
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = JSON.parse(text) as SearchResponse;
      setResults(data.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Florida MLS semantic search
        </h1>
        <p className="text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
          Search by meaning over listing descriptions. Powered by a FastAPI
          backend (embeddings + Chroma). Set{" "}
          <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">
            NEXT_PUBLIC_SEARCH_API_URL
          </code>{" "}
          for production.
        </p>
      </header>

      <form onSubmit={onSearch} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1 text-sm font-medium text-zinc-800 dark:text-zinc-200">
          Query
          <textarea
            className="min-h-[88px] rounded-lg border border-zinc-300 bg-white px-3 py-2 text-base text-zinc-900 shadow-sm outline-none focus:border-zinc-500 dark:border-zinc-600 dark:bg-zinc-950 dark:text-zinc-100"
            placeholder="e.g. screened pool renovated kitchen gated community"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
        </label>
        <label className="flex max-w-xs flex-col gap-1 text-sm font-medium text-zinc-800 dark:text-zinc-200">
          Results (k)
          <input
            type="number"
            min={1}
            max={20}
            className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-950 dark:text-zinc-100"
            value={k}
            onChange={(e) => setK(Number(e.target.value) || 5)}
            disabled={loading}
          />
        </label>
        <button
          type="submit"
          disabled={loading}
          className="w-fit rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error ? (
        <p
          className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200"
          role="alert"
        >
          {error}
        </p>
      ) : null}

      {results.length > 0 ? (
        <ol className="flex flex-col gap-6">
          {results.map((hit, i) => (
            <li
              key={hit.id}
              className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950"
            >
              <div className="mb-2 flex flex-wrap items-baseline gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                  {i + 1}. {hit.id}
                </span>
                {hit.similarity != null && hit.distance != null ? (
                  <span>
                    similarity ≈ {hit.similarity.toFixed(3)} (distance{" "}
                    {hit.distance.toFixed(4)})
                  </span>
                ) : null}
              </div>
              <dl className="mb-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <div>
                  <dt className="text-zinc-500 dark:text-zinc-400">Sold</dt>
                  <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                    {String(hit.metadata.lastSoldPrice ?? "—")}
                  </dd>
                </div>
                <div>
                  <dt className="text-zinc-500 dark:text-zinc-400">ZIP</dt>
                  <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                    {String(hit.metadata.zip ?? "—")}
                  </dd>
                </div>
                <div>
                  <dt className="text-zinc-500 dark:text-zinc-400">Type</dt>
                  <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                    {String(hit.metadata.type ?? "—")}
                  </dd>
                </div>
                <div>
                  <dt className="text-zinc-500 dark:text-zinc-400">Sqft</dt>
                  <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                    {String(hit.metadata.sqft ?? "—")}
                  </dd>
                </div>
              </dl>
              <p className="text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
                {hit.text.length > 1200 ? `${hit.text.slice(0, 1200)}…` : hit.text}
              </p>
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
