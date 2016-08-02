import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { classifySample, classifyMessage, declassifySample, saveSample } from '../actions'
import Sample from '../components/Sample'
import { getSample as getSampleReducer } from '../reducers/samples'
import { byId as recordingsReducer } from '../reducers/recordings'
import { getSamples } from '../actions'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'
import SampleVideo from '../components/SampleVideo'

class SamplesContainer extends Component {

  constructor() {
    super()
    this.state = {
      selected_message: null
    }
    this.handleMessageClick = this.handleMessageClick.bind(this)
    this.handleClassifyClick = this.handleClassifyClick.bind(this)
  }

  handleMessageClick(message_id) {
    this.setState({selected_message: message_id})
  }

  handleClassifyClick(label) {
    if(this.state.selected_message) {
       this.props.classifyMessage(this.state.selected_message, label)
       this.setState({selected_message: null})
    } else {
      this.props.classifySample(this.props.sample, label)
    }
  }

  componentDidMount() {
    this.props.getSamples(this.props.params.recording_id, this.props.params.interval)
  }

  componentWillReceiveProps(nextProps) {
    if (nextProps.params.interval !== this.props.params.interval) {
      this.props.getSamples(this.props.params.recording_id, this.props.params.interval)
      this.setState({selected_message: null})
    }
  }

  render() {
    const { sample, params, recording } = this.props

    return (
      <div>
            <Col xs={12} sm={7} md={7}>
              <SampleVideo video_id={recording.video_id} sample={sample} />
            </Col>
            <Col xs={12} sm={5} md={5} className='hspace'>
              <Col className='hspace'>
                <SampleToolbar
                  recording_id={params.recording_id}
                  interval={params.interval}
                  onReloadClicked={() => this.props.getSamples(params.recording_id, params.interval)}
                  onClassifyClicked={this.handleClassifyClick}
                  onUndoClicked={() => this.props.declassifySample(sample)}
                  onSaveClicked={() => this.props.saveSample(sample, params.recording_id, params.interval)} />
              </Col>
              <Col>
              {sample &&
                <Sample
                  sample={sample}
                  onMessageClicked={(message_id) => { this.handleMessageClick(message_id) }}
                  selected_message={this.state.selected_message} />
              }
              </Col>
            </Col>
      </div>
    )
  }
}

SamplesContainer.propTypes = {
  sample: PropTypes.any,
  recording: PropTypes.any,
  params: PropTypes.any,
  classifyMessage: PropTypes.func.isRequired,
  classifySample: PropTypes.func.isRequired,
  getSamples: PropTypes.func.isRequired,
  declassifySample: PropTypes.func.isRequired,
  saveSample: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    sample: getSampleReducer(state, ownProps.params.recording_id),
    recording: recordingsReducer(state, ownProps.params.recording_id)
  }
}

export default connect(
  mapStateToProps,
  {
    classifySample,
    getSamples,
    classifyMessage,
    declassifySample,
    saveSample
  }
)(SamplesContainer)
