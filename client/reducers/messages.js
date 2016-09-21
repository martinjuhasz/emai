import { RECEIVE_MESSAGE, RECEIVE_MESSAGES, CLASSIFY_MESSAGE_NEW, DECLASSIFY_MESSAGES } from '../constants/ActionTypes'
import merge from 'lodash/merge'

function messages(state = {}, action) {
  switch (action.type) {
    case RECEIVE_MESSAGE:
      return Object.assign({}, state, messagesToHash([action.message]))
    case RECEIVE_MESSAGES:
      return Object.assign({}, state, messagesToHash(action.messages))
    case CLASSIFY_MESSAGE_NEW:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: action.label})
      })
    case DECLASSIFY_MESSAGES:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: null})
      })
    default:
      return state
  }
}

export function byId(state, ids) {
  const messages = []
  for (const id of ids) {
    if (state.messages[id]) {
      messages.push(state.messages[id])
    }
  }
  return messages
}

export function byRecording(state, recording_id) {
  const messages = []
  /*
  for (const key of Object.keys(state.messages)) {
    if (messages[key].channel_id === ) {

    }
  }
  */
  return messages
}

function messagesToHash(messages) {
  const hash = {}
  for (const message of messages) {
    hash[message.id] = message
  }
  return hash
}

export default messages
