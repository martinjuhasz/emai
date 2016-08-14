import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId, getReviews } from '../reducers/classifiers'
import { getReview, trainClassifier } from '../actions'
import Review from '../components/Review'
import ClassifierResultChart from '../components/ClassifierResultChart'
import ClassifierSettings from '../components/ClassifierSettings'
import { Col, ButtonToolbar, Button } from 'react-bootstrap/lib'

class TrainingContainer extends Component {

  constructor() {
    super()
    this.renderResult = this.renderResult.bind(this)
    this.renderReview = this.renderReview.bind(this)
  }

  render() {
    const { classifier } = this.props
    if (!classifier) { return null }
    return (
      <div>
        <h2>{classifier.title}</h2>

        <ButtonToolbar>
          <Button bsStyle="danger" onTouchTap={() => {this.props.onTrainClassifierClicked()}}>Train</Button>
          <Button onTouchTap={() => this.props.onGetReviewClicked()}>Load Review</Button>
        </ButtonToolbar>

        <ClassifierSettings classifier={classifier} />

        <h3>Performance</h3>
        <Col xs={12} sm={7} md={7}>
          { this.renderResult() }
        </Col>
        <Col xs={12} sm={5} md={5}>
          { this.renderReview() }
        </Col>
      </div>
    )
  }

  renderResult() {
    const { classifier } = this.props
    if(!classifier) {
      return null
    }
    return(
      <ClassifierResultChart classifier={classifier} />
    )
  }

  renderReview() {
    const { classifier, reviews } = this.props
    if(!reviews || !classifier) {
      return null
    }
    return(
      <Review messages={reviews} classifier={classifier} />
    )
  }
}

TrainingContainer.propTypes = {
  classifier: PropTypes.any,
  reviews: PropTypes.any,
  onGetReviewClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    classifier: byId(state, ownProps.params.classifier_id),
    reviews: getReviews(state, ownProps.params.classifier_id)
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    onGetReviewClicked: () => {
      dispatch(getReview(ownProps.params.classifier_id))
    },
    onTrainClassifierClicked: () => {
      dispatch(trainClassifier(ownProps.params.classifier_id))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingContainer)
