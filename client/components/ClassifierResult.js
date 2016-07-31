import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'


class ClassifierResult extends Component {
  render() {
    const { result, title } = this.props

    return (
      <div>
        <h4>{title}</h4>
        precision: {result.precision}
        fscore: {result.fscore}
        recall: {result.recall}
        support: {result.support}
      </div>
    )
  }
}

ClassifierResult.propTypes = {
  result: PropTypes.any.isRequired
}


export default connect(

)(ClassifierResult)
