import React, {Component, PropTypes} from 'react'
import {connect} from 'react-redux'
import { getReviews } from '../reducers/classifiers'
import Review from '../components/Review'
import { getReview } from '../actions'
import ClassifierResultChart from '../components/ClassifierResultChart'
import { Col } from 'react-bootstrap/lib'


class ClassifierPerformance extends Component {

  constructor() {
    super()
    this.renderResult = this.renderResult.bind(this)
    this.renderReview = this.renderReview.bind(this)
  }

  componentDidMount() {
    /*
    if(!this.props.reviews && this.props.classifier && this.props.classifier.performance) {
      this.props.getReview(this.props.classifier.id)
    }
    */
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
  reviews: PropTypes.any,
  getReview: PropTypes.func.isRequired
}


function mapStateToProps(state, ownProps) {
  return {
    reviews: getReviews(state, ownProps.classifier.id)
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    getReview: () => {
      dispatch(getReview(ownProps.classifier.id))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(ClassifierPerformance)
