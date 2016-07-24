import 'babel-polyfill'
import React from 'react'
import { render } from 'react-dom'
import { createStore, applyMiddleware } from 'redux'
import { Provider } from 'react-redux'
import logger from 'redux-logger'
import thunk from 'redux-thunk'
import reducer from './reducers'
import { getSamples, getRecordings } from './actions'
import App from './containers/App'
import injectTapEventPlugin from 'react-tap-event-plugin';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';

// http://stackoverflow.com/a/34015469/988941
injectTapEventPlugin();

const middleware = process.env.NODE_ENV === 'production' ?
  [ thunk ] :
  [ thunk, logger() ]

const store = createStore(
  reducer,
  applyMiddleware(...middleware)
)

store.dispatch(getRecordings())

render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('root')
)
