import emai from '../api/emai'
import * as types from '../constants/ActionTypes'
import { getSamples as getSamplesReducer, getUnlabeledMessages, getMessages } from '../reducers/samples'

function receiveSamples(recording_id, samples) {
  return {
    type: types.RECEIVE_SAMPLES,
    samples: samples,
    recording_id: recording_id
  }
}

export function getSamples(recording_id, interval) {
  return dispatch => {
    emai.getSamples(recording_id, interval, samples => {
      dispatch(receiveSamples(recording_id, samples))
    })
  }
}

export function classifySample(sample, label) {
  return (dispatch, getState) => {
    const state = getState()
    const messages = getUnlabeledMessages(state, sample.messages)
    for(const message of messages) {
      dispatch({
        type: types.CLASSIFY_MESSAGE,
        message: message._id,
        label: label
      })
    }
  }
}

export function saveSample(sample, recording_id, interval) {
  return (dispatch, getState) => {

    dispatch({
      type: types.CLASSIFY_SAMPLE,
      sample: sample,
      recording_id: recording_id
    })

    const state = getState()
    const messages = getMessages(state, sample.messages)
    emai.classifyMessages(messages)

    checkForNewSamples(dispatch, state, recording_id, interval)
  }
}

export function declassifySample(sample) {
  return (dispatch, getState) => {
    const state = getState()
    const messages = getMessages(state, sample.messages)
    for(const message of messages) {
      dispatch({
        type: types.DECLASSIFY_MESSAGE,
        message: message._id,
      })
    }
  }
}

export function classifyMessage(message, label) {
  return {
    type: types.CLASSIFY_MESSAGE,
    message: message,
    label: label
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

function checkForNewSamples(dispatch, state, recording_id ,interval) {
  const samples = getSamplesReducer(state, recording_id)
  if (!samples || (samples && samples.length <= 0)) {
    dispatch(getSamples(recording_id, interval))
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
