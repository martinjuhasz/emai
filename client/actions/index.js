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

export function getClassifiers() {
  return dispatch => {
    emai.getClassifiers(classifiers => {
      dispatch(receiveClassifiers(classifiers))
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

export function getReview(classifier_id) {
  return dispatch => {
    emai.getReview(classifier_id, reviews => {
      dispatch(receiveReview(classifier_id, reviews))
    })
  }
}

function receiveReview(classifier_id, reviews) {
  return {
    type: types.RECEIVE_REVIEW,
    reviews: reviews,
    classifier: classifier_id
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

export function saveReview(classifier, messages) {
  return (dispatch, getState) => {
    dispatch({
      type: types.SAVE_REVIEW,
      messages: messages,
      classifier: classifier
    })
    emai.classifyMessages(messages, () => {
      checkForNewReviews(dispatch, getState(), classifier)
    })
  }
}

function checkForNewReviews(dispatch, state, classifier) {
  const reviews = getReviewsReducer(state, classifier)
  if (!reviews || (reviews && reviews.length <= 0)) {
    dispatch(getReview(classifier))
    dispatch(trainClassifier(classifier))

  }
}

export function trainClassifier(classifier_id) {
  return dispatch => {
    emai.trainClassifier(classifier_id, classifier => {
      dispatch(receiveClassifier(classifier))
    })
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

export function updateClassifier(classifier_id, settings, type) {
  return dispatch => {
    emai.updateClassifier(classifier_id, settings, type, () => {
      console.log('updated')
    })
  }
}
