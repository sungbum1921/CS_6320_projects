// src/sagas.js
import { takeEvery, put, delay } from 'redux-saga/effects';
import { increment, decrement, setLoading } from '../slices/login.slice';

function* incrementAsync() {
    yield put(setLoading(true));
    yield delay(1000);
    yield put(increment());
    yield put(setLoading(false));
}

function* decrementAsync() {
    yield put(setLoading(true));
    yield delay(1000);
    yield put(decrement());
    yield put(setLoading(false));
}

export function* watchIncrementAsync() {
    yield takeEvery('counter/incrementAsync', incrementAsync);
}

export function* watchDecrementAsync() {
    yield takeEvery('counter/decrementAsync', decrementAsync);
}