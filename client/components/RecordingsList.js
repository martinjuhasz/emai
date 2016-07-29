import React, { Component, PropTypes } from 'react'

export default class RecordingsList extends Component {
  render() {
    return (
      <div>
        <h3>{this.props.title}</h3>
        <div>{this.props.children}</div>
      </div>
    )
  }
}

RecordingsList.propTypes = {
  children: PropTypes.node
}
