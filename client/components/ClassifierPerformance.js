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
        <Col xs={12} sm={12} md={12}>
          <h3>Performance</h3>
        </Col>
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
      <div className="lhspace">
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits">5671</div>
            <div className="description">total</div>
          </Panel>
        </Col>
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits">1815</div>
            <div className="description">unlabeled</div>
          </Panel>
        </Col>
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits">3856</div>
            <div className="description">labeled</div>
          </Panel>
        </Col>
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits success">1949</div>
            <div className="description">positiv</div>
          </Panel>
        </Col>
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits danger">445</div>
            <div className="description">negativ</div>
          </Panel>
        </Col>
        <Col xs={4} sm={4} md={4}>
          <Panel className="stats">
            <div className="digits warning">1462</div>
            <div className="description">neutral</div>
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
