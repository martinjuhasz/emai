import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { selectRecording } from '../actions'
import Recording from '../components/Recording'
import RecordingsList from '../components/RecordingsList'

class RecordingsContainer extends Component {
  render() {
    const { recordings } = this.props
    return (
      <RecordingsList title="Samples">
        {recordings.map(recording =>
          <Recording
            key={recording.id}
            recording={recording}
            onSelectRecordingClicked={() => { this.props.selectRecording(recording.id) }} />
        )}
      </RecordingsList>
    )
  }
}

RecordingsContainer.propTypes = {
  recordings: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired
  })).isRequired,
  selectRecording: PropTypes.func.isRequired
}

function mapStateToProps(state) { 
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps,
  { selectRecording }
)(RecordingsContainer)
