import { call, put, takeLatest } from 'redux-saga/effects';
import axios from 'axios';
import { queryRequest, querySuccess, queryFailure } from '../slices/query.slice';
import { QueryInput, QueryResponse } from '../../types/query';

function* handleQueryRequest(action: ReturnType<typeof queryRequest>) {
  try {
    const response: { data: QueryResponse } = yield call(
      axios.post,
      'http://localhost:8000/query/',
      { query: action.payload.query, model_id: action.payload.model_id }
    );
    console.log(response.data)
    yield put(querySuccess(response.data));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    yield put(queryFailure(message));
  }
}

export function* querySaga() {
  yield takeLatest(queryRequest.type, handleQueryRequest);
}
