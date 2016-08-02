import React, { Component, PropTypes } from 'react'
import Video from '../components/Video'
import VideoToolbar from '../components/VideoToolbar'
import {Col, Panel} from 'react-bootstrap/lib'

export default class SampleVideo extends Component {

  constructor() {
    super()

    this.timeFrameBefore = 25
    this.timeFrameAfter = 10
    this.handlePlayClick = this.handlePlayClick.bind(this)
    this.handleStopClick = this.handleStopClick.bind(this)
  }

  handlePlayClick() {
    this.refs.video.stop()
    this.refs.video.seek(Math.max(this.props.sample.video_start - this.timeFrameBefore, 0))
    this.refs.video.play()
  }

  handleStopClick() {
    this.refs.video.stop()
  }

  render() {
    const { video_id, sample } = this.props
    if (!sample) {
      return null
    }

    return (
      <Col>
        <Col className='hspace'>
          <VideoToolbar
            onPlayClicked={this.handlePlayClick}
            onStopClicked={this.handleStopClick}/>
        </Col>
        <Col>
          <Panel>
            <Video video_id={video_id} stop_time={sample.video_start + this.timeFrameAfter} ref='video'/>
          </Panel>
        </Col>
      </Col>
    )
  }
}

SampleVideo.propTypes = {
  sample: PropTypes.any,
  video_id: PropTypes.string.isRequired
}
