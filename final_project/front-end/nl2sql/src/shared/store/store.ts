import { configureStore } from '@reduxjs/toolkit';
import createSagaMiddleware from 'redux-saga';
import queryReducer from '../slices/query.slice';
import { querySaga } from '../sagas/query.saga';

const sagaMiddleware = createSagaMiddleware();

export const store = configureStore({
  reducer: {
    query: queryReducer
  },
  middleware: (getDefaultMiddleware) => 
    getDefaultMiddleware().concat(sagaMiddleware) // Use concat() instead of array spread
});

sagaMiddleware.run(querySaga);

// Type-safe hooks
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;