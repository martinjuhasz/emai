import React, { Component } from 'react'
import SamplesContainer from './SamplesContainer'
import RecordingsContainer from './RecordingsContainer'

export default class App extends Component {
  render() {
    return (
      <div>
        <h2>Test UI</h2>
        <hr/>
        <RecordingsContainer />
        <SamplesContainer />
      </div>
    )
  }
}
