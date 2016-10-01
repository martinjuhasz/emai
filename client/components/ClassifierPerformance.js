import React, {Component, PropTypes} from 'react'
import {connect} from 'react-redux'
import ClassifierResultChart from '../components/ClassifierResultChart'
import { Col } from 'react-bootstrap/lib'


class ClassifierPerformance extends Component {

  constructor() {
    super()
    this.renderResult = this.renderResult.bind(this)
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
}

ClassifierPerformance.propTypes = {
  classifier: PropTypes.any
}

export default connect(
)(ClassifierPerformance)
