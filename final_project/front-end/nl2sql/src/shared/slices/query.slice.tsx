import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { QueryState, QueryInput, QueryResponse } from '../../types/query';

const initialState: QueryState = {
  data: null,
  loading: false,
  error: null
};

const querySlice = createSlice({
  name: 'query',
  initialState,
  reducers: {
    queryRequest: (state, action: PayloadAction<QueryInput>) => {
      state.loading = true;
      state.error = null;
    },
    querySuccess: (state, action: PayloadAction<QueryResponse>) => {
      state.loading = false;
      state.data = action.payload;
    },
    queryFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    }
  }
});

export const { queryRequest, querySuccess, queryFailure } = querySlice.actions;
export default querySlice.reducer;
