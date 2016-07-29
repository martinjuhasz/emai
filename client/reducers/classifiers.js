import { combineReducers } from 'redux'
import { RECEIVE_CLASSIFIERS } from '../constants/ActionTypes'
import merge from 'lodash/merge'

export function classifiers(state = {}, action) {
  if(action && action.type === RECEIVE_CLASSIFIERS && action.classifiers) {
    return merge({}, state, {
      [action.recording_id]: action.classifiers
    })
  }
  return state;
}

export function byRecording(state, record_id) {
  return state.classifiers[record_id]
}


export default classifiers