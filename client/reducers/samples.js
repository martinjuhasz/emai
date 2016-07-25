import { combineReducers } from 'redux'
import { CLASSIFY_SAMPLE, RECEIVE_SAMPLES, CHECK_MESSAGE } from '../constants/ActionTypes'
import merge from 'lodash/merge'

function byRecording(state = {}, action) {
  if(action && action.type === RECEIVE_SAMPLES && action.samples && action.samples.result) {
    /*
    return merge({}, state, {
      [action.recording_id]: action.samples.result
    })
    */
    return {
      [action.recording_id]: action.samples.result
    }
  }
  return state;
}

function samples(state = {}, action) {
  switch(action.type) {
    case RECEIVE_SAMPLES:
      //return merge({}, state, action.samples.entities.sample)
      return action.samples.entities.sample
    case CLASSIFY_SAMPLE:
      return merge({}, state, {
        [action.sample.id]: merge({}, state[action.sample.id], {classified: true})
      })
    default:
      return state
  }
}

function messages(state = {}, action) {
  switch(action.type) {
    case RECEIVE_SAMPLES:
      //return merge({}, state, action.samples.entities.message)
      return action.samples.entities.message
    case CHECK_MESSAGE:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {hidden: true})
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

export function getMessages(state, message_ids) {
  return message_ids.map(mid => state.samples.messages[mid]).filter(message => !message.hidden)
}

export function filterHiddenMessages(state, message_ids) {
  return message_ids.map(mid => state.samples.messages[mid]).filter(message => message.hidden).map(m => m._id)
}

export default combineReducers({
  byRecording,
  samples,
  messages
})