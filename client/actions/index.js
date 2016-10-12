import emai from '../api/emai'
import * as types from '../constants/ActionTypes'
import { getSamples as getSamplesReducer, getUnlabeledMessages, getMessages } from '../reducers/samples'
import { getReviews as getReviewsReducer } from '../reducers/classifiers'

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
    emai.classifyMessages(messages, () => {
      checkForNewSamples(dispatch, state, recording_id, interval)
    })
  }
}

export function declassifySample(sample) {
  return (dispatch, getState) => {
    const state = getState()
    const messages = getMessages(state, sample.messages)
    for(const message of messages) {
      dispatch({
        type: types.DECLASSIFY_MESSAGE,
        message: message._id
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

function checkForNewSamples(dispatch, state, recording_id ,interval) {
  const samples = getSamplesReducer(state, recording_id)
  if (!samples || (samples && samples.length <= 0)) {
    dispatch(getSamples(recording_id, interval))
  }
}

export function getClassifiers() {
  return dispatch => {
    emai.getClassifiers(classifiers => {
      dispatch(receiveClassifiers(classifiers))
    })
  }
}

export function deleteClassifier(classifier_id) {
  return dispatch => {
    emai.deleteClassifier(classifier_id, () => {
      dispatch(getClassifiers())
    })
  }
}

export function createClassifier(title) {
  return dispatch => {
    emai.createClassifier(title, () => {
      dispatch(getClassifiers())
    })
  }
}

function receiveClassifiers(classifiers) {
  return {
    type: types.RECEIVE_CLASSIFIERS,
    classifiers: classifiers
  }
}

function receiveClassifier(classifier) {
  return {
    type: types.RECEIVE_CLASSIFIER,
    classifier: classifier
  }
}

export function trainClassifier(classifier_id, limit) {
  return dispatch => {
    emai.trainClassifier(classifier_id, limit, classifier => {
      dispatch(receiveClassifier(classifier))
    })
  }
}

export function updateClassifier(classifier_id, settings, type) {
  return dispatch => {
    emai.updateClassifier(classifier_id, settings, type, classifier => {
      dispatch(receiveClassifier(classifier))
    })
  }
}

export function learnClassifier(classifier_id) {
  return dispatch => {
    emai.learnClassifier(classifier_id, payload => {
      dispatch(receiveClassifier(payload.classifier))
      dispatch(receiveMessages(payload.messages))
    })
  }
}

export function classifyReview(messages, label) {
  return (dispatch, getState) => {
    const unlabeled_messages = messages.filter(message => { return (!message.label || message.label <= 0)})
    for(const message of unlabeled_messages) {
      dispatch({
        type: types.CLASSIFY_REVIEW,
        message: message.id,
        label: label
      })
    }
  }
}

export function declassifyReview(messages) {
  return (dispatch, getState) => {
    const state = getState()
    for(const message of messages) {
      dispatch({
        type: types.DECLASSIFY_REVIEW,
        message: message.id
      })
    }
  }
}

export function classifyReviewMessage(message, label) {
  return {
    type: types.CLASSIFY_REVIEW,
    message: message,
    label: label
  }
}

function receiveMessages(messages) {
  return {
    type: types.RECEIVE_MESSAGES,
    messages: messages
  }
}
