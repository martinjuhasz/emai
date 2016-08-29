import * as types from '../constants/ActionTypes'
import emai from '../api/emai'
import { learnClassifier } from './index'

export function classify(message, label) {
  return {
    type: types.CLASSIFY_MESSAGE_NEW,
    message: message,
    label: label
  }
}

export function classifyUnlabeled(messages, label) {
  return (dispatch) => {
    const unlabeled_messages = messages.filter(message => { return (!message.label || message.label <= 0)})
    for(const message of unlabeled_messages) {
      dispatch({
        type: types.CLASSIFY_MESSAGE_NEW,
        message: message.id,
        label: label
      })
    }
  }
}

export function declassify(messages) {
  return (dispatch) => {
    for(const message of messages) {
      dispatch({
        type: types.DECLASSIFY_MESSAGES,
        message: message.id
      })
    }
  }
}

export function save(classifier, messages) {
  return (dispatch) => {
    dispatch({
      type: types.SAVE_MESSAGES,
      messages: messages,
      classifier: classifier
    })
    emai.classifyMessages(messages, () => {
      dispatch(learnClassifier(classifier))
    })
  }
}

