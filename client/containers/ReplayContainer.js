import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byRecording as messagesReducer } from '../reducers/messages'
import { byId as recordingsReducer } from '../reducers/recordings'
import { getMessagesAtTime } from '../actions/messages'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'
import Video from '../components/Video'
import {ListGroup, ButtonToolbar, ButtonGroup, DropdownButton, MenuItem, Panel } from 'react-bootstrap/lib'
import MessageGroupItem from '../components/MessageGroupItem'
import { last, takeRight } from 'lodash/array'
import { getClassifiers } from '../actions'

class ReplayContainer extends Component {

  constructor() {
    super()
    this.state = {
      video_time: null,
      messages: [],
      selected_classifier: null
    }
    this.videoTimeUpdated = this.videoTimeUpdated.bind(this)
    this.videoSeeked = this.videoSeeked.bind(this)
    this.classifierSelected = this.classifierSelected.bind(this)
  }

  componentDidMount() {
    this.props.getClassifiers()
  }

  videoTimeUpdated(time) {
    if (time % 2 === 0) {
      const last_message = last(this.state.messages)
      const classifier = (this.state.selected_classifier) ? this.state.selected_classifier.id : false
      getMessagesAtTime(this.props.recording.id, time + 2, last_message, classifier, messages => {
        this.setState({ messages: [...this.state.messages, ...messages] })
      })
    }
    this.setState({video_time: time})
  }

  videoSeeked() {
    this.setState({messages: []})
  }

  classifierSelected(classifier) {
    if(!classifier) {
      this.setState({selected_classifier: null})
      return
    }
    this.setState({selected_classifier: classifier})
  }

  render() {
    const { recording, classifiers } = this.props
    const { messages } = this.state

    return (
      <Col>
        <Col xs={12} sm={7} md={7} className='lhspace'>
          <Panel>
            <Video video_id={recording.video_id} ref='video' onTimeUpdate={this.videoTimeUpdated} onSeeked={this.videoSeeked} controls={true} autoplay={true}/>
          </Panel>

          <div className="hspace">
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits">0000</div>
                <div className="description">total</div>
              </Panel>
            </Col>
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits">0000</div>
                <div className="description">unlabeled</div>
              </Panel>
            </Col>
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits">0000</div>
                <div className="description">labeled</div>
              </Panel>
            </Col>
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits success">0000</div>
                <div className="description">correct</div>
              </Panel>
            </Col>
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits danger">0000</div>
                <div className="description">failed</div>
              </Panel>
            </Col>
            <Col xs={4} sm={4} md={4}>
              <Panel className="stats">
                <div className="digits">0000</div>
                <div className="description">unknown</div>
              </Panel>
            </Col>
          </div>

        </Col>
        <Col xs={12} sm={5} md={5}>
          <Col className='hspace'>
            <ButtonToolbar>
              <ButtonGroup>
                <DropdownButton id='replay_classifier_dropdown' title={(this.state.selected_classifier && this.state.selected_classifier.title) || 'Klassifikatoren'} onSelect={this.classifierSelected}>
                  <MenuItem eventKey={false}>ohne</MenuItem>
                  {classifiers.map(classifier =>
                    <MenuItem key={classifier.id} eventKey={classifier}>{classifier.title}</MenuItem>
                  )}
                </DropdownButton>
              </ButtonGroup>
            </ButtonToolbar>
          </Col>
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
      </Col>
    )
  }
}

ReplayContainer.propTypes = {
  messages: PropTypes.any,
  recording: PropTypes.any,
  classifiers: PropTypes.any,
  params: PropTypes.any,
  getClassifiers: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    recording: recordingsReducer(state, ownProps.params.recording_id),
    classifiers: state.classifiers.all
  }
}

export default connect(
  mapStateToProps,
  {
    getClassifiers
  }
)(ReplayContainer)
