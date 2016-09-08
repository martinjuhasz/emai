import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byRecording as messagesReducer } from '../reducers/messages'
import { byId as recordingsReducer } from '../reducers/recordings'
import { getMessagesAtTime } from '../actions/messages'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'
import Video from '../components/Video'

class ReplayContainer extends Component {

  constructor() {
    super()
    this.state = {
      video_time: null
    }
    this.videoTimeUpdated = this.videoTimeUpdated.bind(this)
  }

  videoTimeUpdated(time) {
    if (time % 2 === 0) {
      this.props.getMessagesAtTime(this.props.recording.id, time + 10)
    }
    this.setState({video_time: time})
  }

  render() {
    const { params, recording } = this.props

    return (
      <div>
            <Col xs={12} sm={7} md={7}>
              <Video video_id={recording.video_id} ref='video' onTimeUpdate={this.videoTimeUpdated} controls={true}/>
            </Col>
            <Col xs={12} sm={5} md={5}>
              <Col>

              </Col>
            </Col>
      </div>
    )
  }
}

ReplayContainer.propTypes = {
  messages: PropTypes.any,
  recording: PropTypes.any,
  params: PropTypes.any,
  getMessagesAtTime: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    messages: messagesReducer(state, ownProps.params.recording_id),
    recording: recordingsReducer(state, ownProps.params.recording_id)
  }
}

export default connect(
  mapStateToProps,
  {
    getMessagesAtTime
  }
)(ReplayContainer)
