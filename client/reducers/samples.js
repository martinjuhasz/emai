import { combineReducers } from 'redux'
import * as types from '../constants/ActionTypes'
import merge from 'lodash/merge'
import unset from 'lodash/unset'
import pull from 'lodash/pull'

function byRecording(state = {}, action) {
  switch(action.type) {
    case types.RECEIVE_SAMPLES:
      return { [action.recording_id]: action.samples.result }
    case types.CLASSIFY_SAMPLE: {
      const new_state = merge({}, state)
      pull(new_state[action.recording_id], action.sample.id)
      return new_state
    }
    default:
      return state
  }
}

function samples(state = {}, action) {
  switch(action.type) {
    case types.RECEIVE_SAMPLES:
      return action.samples.entities.sample
    case types.CLASSIFY_SAMPLE: {
      const new_state = merge({}, state)
      unset(new_state, action.sample.id)
      return new_state
    }
    default:
      return state
  }
}

function messages(state = {}, action) {
  switch(action.type) {
    case types.RECEIVE_SAMPLES:
      return action.samples.entities.message
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
    return sample_ids.map(sid => state.samples.samples[sid]).filter(sample => (sample && !sample.classified))
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
