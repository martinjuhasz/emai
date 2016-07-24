import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Recording from '../components/Recording'
import RecordingsList from '../components/RecordingsList'

const recording_list = (recordings) => {
  return (
      <RecordingsList title="Samples">
        {recordings.map(recording =>
          <Recording
            key={recording.id}
            recording={recording} />
        )}
      </RecordingsList>
    )
}

class RecordingsContainer extends Component {
  render() {
    const { recordings } = this.props
    return (
      <div> {this.props.children || recording_list(recordings)} </div>
    )
  }
}

RecordingsContainer.propTypes = {
  recordings: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired
  })).isRequired
}

function mapStateToProps(state) { 
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps
)(RecordingsContainer)
