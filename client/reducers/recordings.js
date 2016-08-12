import { RECEIVE_RECORDINGS } from '../constants/ActionTypes'


function recordings(state = {all:[]}, action) {
  switch (action.type) {
    case RECEIVE_RECORDINGS:
      return Object.assign({}, state, {
        all: action.recordings
      })
    default:
      return state
  }
}

export function byId(state, record_id) {
  return state.recordings.all.find(record => record.id === record_id)
}

export default recordings
