import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { classifySample, checkMessage, classifyMessage, declassifySample, saveSample } from '../actions'
import Sample from '../components/Sample'
import { getSample as getSampleReducer } from '../reducers/samples'
import { getSamples } from '../actions'
import Video from '../components/Video'
import {Row, Col, Button } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'


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
    console.log(label)
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
    const { sample, params } = this.props

    return (
      <Row>
            <Col xs={12} sm={7} md={7}>
              <Video video_id={params.recording_id} />
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
      </Row>
    )
  }
}

SamplesContainer.propTypes = {
  sample: PropTypes.shape({
    id: PropTypes.string.isRequired,
    messages: PropTypes.any.isRequired
  })
}

function mapStateToProps(state, ownProps) { 
  return {
    sample: getSampleReducer(state, ownProps.params.recording_id)
  }
}

export default connect(
  mapStateToProps,
  {
    classifySample,
    checkMessage,
    getSamples,
    classifyMessage,
    declassifySample,
    saveSample
  }
)(SamplesContainer)
