import React, { Component, PropTypes } from 'react'
import { Link } from 'react-router'
export default class Recording extends Component {
  render() {
    const { recording, path } = this.props
    return (
      <div>
        <Link to={`/${path}/${recording.id}`}>{recording.display_name}</Link>
      </div>
    )
  }
}

Recording.propTypes = {
  recording: PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired
  }),
  path: PropTypes.string.isRequired
}
