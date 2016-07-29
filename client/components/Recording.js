import React, { Component, PropTypes } from 'react'
import { Link } from 'react-router'
import RaisedButton from 'material-ui/RaisedButton';
import Divider from 'material-ui/Divider';

export default class Recording extends Component {
  render() {
    const { recording, path } = this.props
    return (
      <div>
        <RaisedButton label={recording.display_name} primary={true} containerElement={<Link to={`/${path}/${recording.id}`} />} />
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
