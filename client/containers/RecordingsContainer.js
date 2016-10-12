import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Recording from '../components/Recording'
import { getRecordings, deleteRecording, startRecording, stopRecording } from '../actions/recordings'
import { Row, Col, FormControl, Button, Glyphicon } from 'react-bootstrap/lib'

class RecordingsContainer extends Component {

  constructor() {
    super()
    this.state = {
      add_username: null,
    }
    this.renderNewRecordingInput = this.renderNewRecordingInput.bind(this)
    this.renderRecordingList = this.renderRecordingList.bind(this)
    this.usernameInputChanged = this.usernameInputChanged.bind(this)
    this.onStartRecordingClicked = this.onStartRecordingClicked.bind(this)
  }

  componentDidMount() {
    this.props.getRecordings()
  }

  usernameInputChanged(event) {
    this.setState({add_username: event.target.value})
  }

  onStartRecordingClicked() {
    if(this.state.add_username) {
      this.props.startRecording(this.state.add_username)
    }
  }


  renderNewRecordingInput() {
    return (
      <Row>
        <Col xs={5} sm={5} md={5}>
          <FormControl type="text" placeholder="Username" onChange={this.usernameInputChanged} />
        </Col>
        <Col xs={3} sm={3} md={3}>
          <Button onTouchTap={this.onStartRecordingClicked}>
            <Glyphicon glyph="record"/> Record
          </Button>
        </Col>
        <Col xs={4} sm={4} md={4}>

        </Col>
      </Row>
    )
  }

  renderRecordingList(recordings) {
    return (
      <div>
        <h2>Recordings</h2>
        {this.renderNewRecordingInput()}
        <div className="hspace">
          {recordings.map(recording =>
            <Recording
              key={recording.id}
              recording={recording}
              onDeleteClicked={() => this.props.deleteRecording(recording.id)}
              onStopClicked={() => this.props.stopRecording(recording.id)}
              path='recordings' />
          )}
        </div>
      </div>
    )
  }

  render() {
    const { recordings } = this.props
    return (
      <div> {this.props.children || this.renderRecordingList(recordings)} </div>
    )
  }
}

RecordingsContainer.propTypes = {
  recordings: PropTypes.any.isRequired,
  children: PropTypes.node,
  getRecordings: PropTypes.func.isRequired,
  deleteRecording: PropTypes.func.isRequired,
  startRecording: PropTypes.func.isRequired,
  stopRecording: PropTypes.func.isRequired
}

function mapStateToProps(state) {
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps,
  {
    getRecordings,
    deleteRecording,
    startRecording,
    stopRecording
  }
)(RecordingsContainer)
