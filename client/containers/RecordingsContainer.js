import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Recording from '../components/Recording'
import { getRecordings } from '../actions/recordings'

const recording_list = (recordings) => {
  return (
      <div>
        <h2>Recordings</h2>
        {recordings.map(recording =>
          <Recording
            key={recording.id}
            recording={recording}
            path='recordings' />
        )}
      </div>
    )
}

class RecordingsContainer extends Component {

  componentDidMount() {
    this.props.getRecordings()
  }

  render() {
    const { recordings } = this.props
    return (
      <div> {this.props.children || recording_list(recordings)} </div>
    )
  }
}

RecordingsContainer.propTypes = {
  recordings: PropTypes.any.isRequired,
  children: PropTypes.node,
  getRecordings: PropTypes.func.isRequired
}

function mapStateToProps(state) {
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps,
  { getRecordings }
)(RecordingsContainer)
