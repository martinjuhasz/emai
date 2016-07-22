import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { classifySample } from '../actions'
import Sample from '../components/Sample'
import SampleList from '../components/SampleList'

class SamplesContainer extends Component {
  render() {
    const { samples } = this.props
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
  })).isRequired,
  classifySample: PropTypes.func.isRequired
}

function mapStateToProps(state) { 
  return {
    samples: state.samples
  }
}

export default connect(
  mapStateToProps,
  { classifySample }
)(SamplesContainer)
