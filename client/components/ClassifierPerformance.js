import React, {Component, PropTypes} from 'react'
import {connect} from 'react-redux'
import { getReviews } from '../reducers/classifiers'
import Review from '../components/Review'
import ClassifierResultChart from '../components/ClassifierResultChart'
import { Col } from 'react-bootstrap/lib'


class ClassifierPerformance extends Component {

  constructor() {
    super()
    this.renderResult = this.renderResult.bind(this)
    this.renderReview = this.renderReview.bind(this)
  }

  render() {
    const {classifier} = this.props
    if(!classifier || !classifier.performance) {
      return null
    }

    return (
      <div>
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

ClassifierPerformance.propTypes = {
  classifier: PropTypes.any,
  reviews: PropTypes.any
}


function mapStateToProps(state, ownProps) {
  return {
    reviews: getReviews(state, ownProps.classifier.id)
  }
}

export default connect(
  mapStateToProps
)(ClassifierPerformance)
