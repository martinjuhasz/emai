import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { classifySample } from '../actions'
import Sample from '../components/Sample'
import SampleList from '../components/SampleList'
import { getSample } from '../reducers/samples'


class SamplesContainer extends Component {
  render() {
    const { samples } = this.props
    if (!samples || samples.length < 1) { return null }

    return (
      <SampleList title="Samples">
        {samples.map(sample =>
          <Sample
            key={sample.id}
            sample={sample}
            onClassifySampleClicked={(label) => { this.props.classifySample(sample.id, label) }} />
        )}
      </SampleList>
    )
  }
}

SamplesContainer.propTypes = {
  samples: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    messages: PropTypes.any.isRequired,
    time: PropTypes.string.isRequired,
    words: PropTypes.number.isRequired,
    recording_id: PropTypes.string.isRequired,
  })),
  classifySample: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) { 
  return {
    samples: getSample(state, ownProps.recording.id)
  }
}

export default connect(
  mapStateToProps,
  { classifySample }
)(SamplesContainer)
