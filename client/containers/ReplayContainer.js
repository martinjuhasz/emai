import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byRecording as messagesReducer } from '../reducers/messages'
import { byId as recordingsReducer } from '../reducers/recordings'
import { getMessagesAtTime } from '../actions/messages'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'
import Video from '../components/Video'
import {ListGroup } from 'react-bootstrap/lib'
import MessageGroupItem from '../components/MessageGroupItem'
import { last, takeRight } from 'lodash/array'

class ReplayContainer extends Component {

  constructor() {
    super()
    this.state = {
      video_time: null,
      messages: []
    }
    this.videoTimeUpdated = this.videoTimeUpdated.bind(this)
  }

  videoTimeUpdated(time) {
    if (time % 2 === 0) {
      const last_message = last(this.state.messages)
      getMessagesAtTime(this.props.recording.id, time + 2, last_message, messages => {
        this.setState({ messages: [...this.state.messages, ...messages] })
      })
    }
    this.setState({video_time: time})
  }

  render() {
    const { recording } = this.props
    const { messages } = this.state

    return (
      <div>
            <Col xs={12} sm={7} md={7}>
              <Video video_id={recording.video_id} ref='video' onTimeUpdate={this.videoTimeUpdated} controls={true}/>
            </Col>
            <Col xs={12} sm={5} md={5}>
              <Col>
                <ListGroup>
                  {messages && messages.length > 0 && takeRight(messages, 20).map(message => {
                    const message_id = message._id || message.id
                    return (
                      <MessageGroupItem
                        onTouchTap={() => {}}
                        key={message_id}
                        message={message} />
                    )
                  })}
                </ListGroup>
              </Col>
            </Col>
      </div>
    )
  }
}

ReplayContainer.propTypes = {
  messages: PropTypes.any,
  recording: PropTypes.any,
  params: PropTypes.any
}

function mapStateToProps(state, ownProps) {
  return {
    recording: recordingsReducer(state, ownProps.params.recording_id)
  }
}

export default connect(
  mapStateToProps,
  {
  }
)(ReplayContainer)
