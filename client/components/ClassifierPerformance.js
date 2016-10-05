import React, {Component, PropTypes} from 'react'
import {connect} from 'react-redux'
import ClassifierResultChart from '../components/ClassifierResultChart'
import { Col, Panel, Row } from 'react-bootstrap/lib'


class ClassifierPerformance extends Component {

  constructor() {
    super()
    this.renderResult = this.renderResult.bind(this)
    this.renderStatistics = this.renderStatistics.bind(this)
  }

  render() {
    const {classifier} = this.props
    if(!classifier || !classifier.performance) {
      return null
    }

    return (
      <Row>
        <h3>Performance</h3>
        <Col xs={12} sm={7} md={7}>
          { this.renderResult() }
        </Col>
        <Col xs={12} sm={5} md={5}>
          { this.renderStatistics() }
        </Col>
      </Row>
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

  renderStatistics() {
    return (
      <div className="hspace">
        <Col xs={6} sm={6} md={6}>
          <Panel>
            asd
          </Panel>
        </Col>
        <Col xs={6} sm={6} md={6}>
          <Panel>
            asd
          </Panel>
        </Col>
      </div>
    )
  }
}

ClassifierPerformance.propTypes = {
  classifier: PropTypes.any
}

export default connect(
)(ClassifierPerformance)
