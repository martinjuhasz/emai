import { combineReducers } from 'redux'
import { CLASSIFY_SAMPLE, RECEIVE_SAMPLES } from '../constants/ActionTypes'


function byRecording(state = {}, action) {
  switch (action.type) {
    case RECEIVE_SAMPLES:
      return Object.assign({},
        state,
        {
        	[action.recording_id]: action.samples
        }, {})
    default:
      return state
  }
}

export function getSample(state, record_id) {
  return state.samples.byRecording[record_id]
}


export default combineReducers({
  byRecording
})