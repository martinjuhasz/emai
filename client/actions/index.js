import emai from '../api/emai'
import * as types from '../constants/ActionTypes'
import { getSamples as getSamplesReducer, filterHiddenMessages } from '../reducers/samples'

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

export function classifySample(sample, label) {
  return (dispatch, getState) => {
    const state = getState()
    const hiddenMessages = filterHiddenMessages(state, sample.messages)
    emai.classifySample(sample.id, label, hiddenMessages)
    checkForNewSamples(state, sample, dispatch)
    dispatch({
      type: types.CLASSIFY_SAMPLE,
      sample: sample
    })
  }
}

export function checkMessage(sample, message_id) {
  console.log(sample, message_id, types.CHECK_MESSAGE)
  return {
      type: types.CHECK_MESSAGE,
      sample: sample,
      message: message_id
    }
}

function checkForNewSamples(state, sample, dispatch) {
  const samples = getSamplesReducer(state, sample.recording_id)
  if (!samples || (samples && samples.length <= 1)) {
    dispatch(getSamples(sample.recording_id, sample.data_set))
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

export function getClassifiers(recording_id) {
  return dispatch => {
    emai.getClassifiers(recording_id, classifiers => {
      dispatch(receiveClassifiers(recording_id, classifiers))
    })
  }
}

function receiveClassifiers(recording_id, classifiers) {
  return {
    type: types.RECEIVE_CLASSIFIERS,
    classifiers: classifiers,
    recording_id: recording_id
  }
}
