import { combineReducers } from 'redux'
import { RECEIVE_RECORDINGS, SELECT_RECORDING } from '../constants/ActionTypes'


function recordings(state = {all:[]}, action) {
  switch (action.type) {
    case RECEIVE_RECORDINGS:
      return Object.assign({}, state, {
        all: action.recordings
      })
    case SELECT_RECORDING:
      return Object.assign({}, state, {
        selected: action.recording_id
      })
    default:
      return state
  }
}

export function byId(state, record_id) {
  return state.recordings.all.find(record => record.id === record_id)
}

export default recordings
