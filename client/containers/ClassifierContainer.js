import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId, getReviews } from '../reducers/classifiers'
import { getReview } from '../actions'
import Review from '../components/Review'

class TrainingContainer extends Component {

  constructor() {
    super()
    this.renderReview = this.renderReview.bind(this)
  }

  render() {
    const { classifier } = this.props
    if (!classifier) { return null }
    return (
      <div>
        <h2>{classifier.title}</h2>
        <div onTouchTap={() => this.props.onGetReviewClicked()}>review Classifier</div>
        { this.renderReview() }
      </div>
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
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingContainer)
