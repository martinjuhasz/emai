import React, { Component, PropTypes } from 'react'
import { ResponsiveEmbed } from 'react-bootstrap/lib'

export default class Video extends Component {

  constructor() {
    super()

    this.onTimeUpdate = this.onTimeUpdate.bind(this)
    this.onSeeked = this.onSeeked.bind(this)
    this.lastUpdate = 0
  }

  videoURL(video_id) {
    return `videos/${video_id}.mp4`
  }

  play() {
    this.refs.video.play()
  }

  seek(time) {
    this.refs.video.currentTime = time
  }

  stop() {
    this.refs.video.pause()
  }

  onTimeUpdate() {
    if(this.props.stop_time && this.props.stop_time <= this.refs.video.currentTime) {
      this.stop()
    }
    const flooredTime = Math.floor(this.refs.video.currentTime)
    if (this.props.onTimeUpdate && flooredTime !== this.lastUpdate) {
      this.lastUpdate = flooredTime
      this.props.onTimeUpdate(flooredTime)
    }
  }

  onSeeked() {
    this.props.onSeeked()
  }

  render() {
    const { video_id, controls, autoplay } = this.props
    if(!video_id) { return null }

    const videoControls = {controls: controls, autoPlay: autoplay}
    return (
      <ResponsiveEmbed a16by9>
        <video ref='video' onTimeUpdate={this.onTimeUpdate} onSeeked={this.onSeeked} {...videoControls}>
          <source src={this.videoURL(video_id)} type='video/mp4' />
        </video>
      </ResponsiveEmbed>
    )
  }
}

Video.propTypes = {
  video_id: PropTypes.string.isRequired,
  stop_time: PropTypes.number,
  onTimeUpdate: PropTypes.func,
  onSeeked: PropTypes.func,
  controls: PropTypes.bool,
  autoplay: PropTypes.bool.isRequired
}

