import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId } from '../reducers/classifiers'
import { getReview } from '../actions'
import ClassifierSettings from '../components/ClassifierSettings'
import { getRecordings } from '../actions/recordings'

class ClassifierDetailsContainer extends Component {

  componentDidMount() {
    this.props.getRecordings()
  }

  render() {
    const { classifier } = this.props
    if (!classifier) { return null }
    return (
      <div>
        <h2>{classifier.title}</h2>

        <ClassifierSettings classifier={classifier} />
      </div>
    )
  }
}

ClassifierDetailsContainer.propTypes = {
  classifier: PropTypes.any,
  getRecordings: PropTypes.func
}

function mapStateToProps(state, ownProps) {
  return {
    classifier: byId(state, ownProps.params.classifier_id),
  }
}

export default connect(
  mapStateToProps,
  {
    getRecordings
  }
)(ClassifierDetailsContainer)
