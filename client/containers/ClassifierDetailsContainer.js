import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId } from '../reducers/classifiers'
import { getReview, learnClassifier } from '../actions'
import ClassifierSettings from '../components/ClassifierSettings'
import ClassifierPerformance from '../components/ClassifierPerformance'
import { ButtonToolbar, Button } from 'react-bootstrap/lib'

class ClassifierDetailsContainer extends Component {

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
  classifier: PropTypes.any
}

function mapStateToProps(state, ownProps) {
  return {
    classifier: byId(state, ownProps.params.classifier_id),
  }
}

export default connect(
  mapStateToProps
)(ClassifierDetailsContainer)
