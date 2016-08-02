import React, { Component, PropTypes } from 'react'
import { ResponsiveEmbed } from 'react-bootstrap/lib'

export default class Video extends Component {

  constructor() {
    super()

    this.onTimeUpdate = this.onTimeUpdate.bind(this)
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
  }

  render() {
    const { video_id } = this.props
    if(!video_id) { return null }

    return (
      <ResponsiveEmbed a16by9>
        <video ref='video' onTimeUpdate={this.onTimeUpdate}>
          <source src={this.videoURL(video_id)} type='video/mp4' />
        </video>
      </ResponsiveEmbed>
    )
  }
}

Video.propTypes = {
  video_id: PropTypes.string.isRequired,
  stop_time: PropTypes.number.isRequired
}

