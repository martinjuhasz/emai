import emai from '../api/emai'
import * as types from '../constants/ActionTypes'

function receiveSamples(samples) {
  return {
    type: types.RECEIVE_SAMPLES,
    samples: samples
  }
}

export function getSamples(recording_id) {
  return dispatch => {
    emai.getSamples(recording_id, samples => {
      dispatch(receiveSamples(samples))
    })
  }
}

export function classifySample(sample_id, label) {
  return dispatch => {
    emai.classifySample(sample_id, label)
    dispatch({
      type: types.CLASSIFY_SAMPLE,
      sample: sample_id
    })
  }
}

function receiveRecordings(recordings) {
  return {
    type: types.RECEIVE_RECORDINGS,
    recordings: recordings
  }
}

export function getRecordings() {
  return dispatch => {
    emai.getRecordings(recordings => {
      dispatch(receiveRecordings(recordings))
    })
  }
}

export function selectRecording(recording_id) {
  return dispatch => {
    dispatch(getSamples(recording_id))
    dispatch({
      type: types.SELECT_RECORDING,
      recording_id: recording_id
    })
  }
}