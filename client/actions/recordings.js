import * as types from '../constants/ActionTypes'
import emai from '../api/emai'

function receiveRecordings(recordings) {
  return {
    type: types.RECEIVE_RECORDINGS,
    recordings: recordings
  }
}

export function getRecordings() {
  return dispatch => {
    emai.getRecordings(recordings => {
      dispatch(receiveRecordings(recordings))
    })
  }
}

export function deleteRecording(recording_id) {
  return dispatch => {
    emai.deleteRecording(recording_id, () => {
      dispatch(getRecordings())
    })
  }
}

export function startRecording(username) {
  return dispatch => {
    emai.startRecording(username, () => {
      dispatch(getRecordings())
    })
  }
}

export function stopRecording(recording_id) {
  return dispatch => {
    emai.stopRecording(recording_id, () => {
      dispatch(getRecordings())
    })
  }
}
