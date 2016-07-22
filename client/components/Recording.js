import React, { Component, PropTypes } from 'react'

export default class Recording extends Component {
  render() {
    const { recording } = this.props
    return (
    	<div onClick={this.props.onSelectRecordingClicked}>{recording.display_name}: {recording.started} - {recording.stopped}</div>
	   )
  }
}

Recording.propTypes = {
  recording: PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired
  }),
  onSelectRecordingClicked: PropTypes.func.isRequired
}
