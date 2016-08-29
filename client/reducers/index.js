import { combineReducers } from 'redux'
import samples from './samples'
import recordings from './recordings'
import classifiers from './classifiers'
import messages from './messages'

export default combineReducers({
  samples,
  recordings,
  classifiers,
  messages
})

