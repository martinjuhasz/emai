import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'

class Classifier extends Component {
  render() {
    const { video_id, title } = this.props
    const video_url = `videos/${video_id}.mp4`

    return (
      <video>
        <source src={video_url} type="video/mp4" />
      </video>
    )
  }
}

Classifier.propTypes = {
  video_id: PropTypes.string.isRequired
}


export default connect(

)(Classifier)
