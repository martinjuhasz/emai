import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId } from '../reducers/recordings'
import Recording from '../components/Recording'
import { Link } from 'react-router'
import { getSamples } from '../actions'
import SamplesContainer from './SamplesContainer'
import Chip from 'material-ui/Chip';

const styles = {
    chip: {
      margin: 4,
    },
    chips: {
      display: 'flex',
      flexWrap: 'wrap',
    },
  };

class RecordingContainer extends Component {

  render() {
    const { recording } = this.props
    if (!recording) { return null }
    
    return (
      <div>
        <h2>{recording.display_name}</h2>
        <div style={styles.chips}>
        { recording.data_sets.sort((a,b) => { return a-b}).map(data_set => 
          <Chip key={data_set} onTouchTap={() => this.props.onDataSetClicked(data_set)} style={styles.chip}>{data_set}</Chip>
        )}
        </div>

        <SamplesContainer recording={recording} />
      </div>
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
  }),
  onDataSetClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) { 
  return {
    recording: byId(state, ownProps.params.recording_id)
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    onDataSetClicked: (data_set) => {
      dispatch(getSamples(ownProps.params.recording_id, data_set))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(RecordingContainer)
