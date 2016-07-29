import { combineReducers } from 'redux'
import samples from './samples'
import recordings from './recordings'
import classifiers from './classifiers'

export default combineReducers({
  samples,
  recordings,
  classifiers
})

