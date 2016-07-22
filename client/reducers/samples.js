import { combineReducers } from 'redux'
import { CLASSIFY_SAMPLE, RECEIVE_SAMPLES } from '../constants/ActionTypes'


function samples(state = [], action) {
  switch (action.type) {
    case RECEIVE_SAMPLES:
      return action.samples
    case CLASSIFY_SAMPLE:
    	
    	return state
    default:
      return state
  }
}

export default samples
