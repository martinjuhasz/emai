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

        <ButtonToolbar>
          <Button bsStyle="danger" onTouchTap={() => {this.props.onTrainClassifierClicked()}}>Learn</Button>
          <Button onTouchTap={() => this.props.onGetReviewClicked()}>Load Review</Button>
        </ButtonToolbar>

        <ClassifierSettings classifier={classifier} />

        <ClassifierPerformance classifier={classifier} />
      </div>
    )
  }
}

ClassifierDetailsContainer.propTypes = {
  classifier: PropTypes.any,
  onGetReviewClicked: PropTypes.func.isRequired,
  onTrainClassifierClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    classifier: byId(state, ownProps.params.classifier_id),
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    onGetReviewClicked: () => {
      dispatch(getReview(ownProps.params.classifier_id))
    },
    onTrainClassifierClicked: () => {
      dispatch(learnClassifier(ownProps.params.classifier_id))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(ClassifierDetailsContainer)
