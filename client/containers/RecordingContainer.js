import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId } from '../reducers/recordings'
import Recording from '../components/Recording'
import { Link } from 'react-router'
import SamplesContainer from './SamplesContainer'
import { LinkContainer } from 'react-router-bootstrap'
import { Button } from 'react-bootstrap/lib'

const header = (recording) => {
  return (
      <h2>{recording.display_name}</h2>
    )
}

const default_content = (recording) => {
  return (
      <div>
        <LinkContainer to={`/recordings/${recording.id}/samples/10`}>
            <Button>Show Samples</Button>
        </LinkContainer>
      </div>
    )
}

class RecordingContainer extends Component {

  render() {
    const { recording } = this.props
    if (!recording) { return null }

    return (
      <div>
      {header(recording)}
      {this.props.children || default_content(recording)} </div>
    )
  }
}

RecordingContainer.propTypes = {
  recording: PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired,
    data_sets: PropTypes.array
  })
}

function mapStateToProps(state, ownProps) { 
  return {
    recording: byId(state, ownProps.params.recording_id)
  }
}



export default connect(
  mapStateToProps
)(RecordingContainer)
