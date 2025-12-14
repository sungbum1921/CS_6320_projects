// src/counterSlice.js
import { createSlice } from '@reduxjs/toolkit';

export const counterSlice = createSlice({
  name: 'counter',
  initialState: {
    value: 0,
    loading: false,
  },
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementAsync: () => {},
    decrementAsync: () => {},
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
  },
});

export const { increment, decrement, incrementAsync, decrementAsync, setLoading } = counterSlice.actions;
export const selectCount = (state: { counter: { value: any; }; }) => state.counter.value;
export const selectLoading = (state: { counter: { loading: any; }; }) => state.counter.loading;
export default counterSlice.reducer;