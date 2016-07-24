import emai from '../api/emai'
import * as types from '../constants/ActionTypes'

function receiveSamples(recording_id, samples) {
  return {
    type: types.RECEIVE_SAMPLES,
    samples: samples,
    recording_id: recording_id
  }
}

export function getSamples(recording_id, data_set) {
  return dispatch => {
    emai.getSamples(recording_id, data_set, samples => {
      dispatch(receiveSamples(recording_id, samples))
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
