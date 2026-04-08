/** Matches FastAPI SearchHit / SearchResponse */

export type SearchHit = {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  distance: number | null;
  similarity: number | null;
};

export type SearchResponse = {
  results: SearchHit[];
};
