import React, { Component, PropTypes } from 'react'
import { Link } from 'react-router'
import RaisedButton from 'material-ui/RaisedButton';
import Divider from 'material-ui/Divider';

export default class Recording extends Component {
  render() {
    const { recording } = this.props
    return (
      <div>
        <RaisedButton label={recording.display_name} primary={true} containerElement={<Link to={`/recordings/${recording.id}`} />} />
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
  })
}
