import { combineReducers } from 'redux'
import * as types from '../constants/ActionTypes'
import merge from 'lodash/merge'

function byRecording(state = {}, action) {
  if(action && action.type === types.RECEIVE_SAMPLES && action.samples && action.samples.result) {
    return {
      [action.recording_id]: action.samples.result
    }
  }
  return state;
}

function samples(state = {}, action) {
  switch(action.type) {
    case types.RECEIVE_SAMPLES:
      return action.samples.entities.sample
    case types.CLASSIFY_SAMPLE:
      return merge({}, state, {
        [action.sample.id]: merge({}, state[action.sample.id], {classified: true})
      })
    default:
      return state
  }
}

function messages(state = {}, action) {
  switch(action.type) {
    case types.RECEIVE_SAMPLES:
      return action.samples.entities.message
    case types.CHECK_MESSAGE:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {hidden: true})
      })
    case types.CLASSIFY_MESSAGE:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: action.label})
      })
    case types.DECLASSIFY_MESSAGE:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: null})
      })
    default:
      return state
  }
}

export function getSamples(state, record_id) {
  const sample_ids = state.samples.byRecording[record_id]
  if(sample_ids) {
    return sample_ids.map(sid => state.samples.samples[sid]).filter(sample => !sample.classified)
  }
}

export function getSample(state, recording_id) {
  const samples = getSamples(state, recording_id)
  return (samples && samples[0]) ? samples[0] : null
}

export function getMessages(state, message_ids) {
  return message_ids.map(mid => state.samples.messages[mid])
}

export function getUnlabeledMessages(state, message_ids) {
  return message_ids.map(mid => state.samples.messages[mid]).filter(message => !('label' in message) || message.label == null)
}

export default combineReducers({
  byRecording,
  samples,
  messages
})