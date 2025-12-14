export interface QueryInput {
  query: string;
  model_id?: string;
}

export interface QueryResponse {
  message: string;
  timestamp: string;
  type: string;
}

export interface QueryState {
  data: QueryResponse | null;
  loading: boolean;
  error: string | null;
}
