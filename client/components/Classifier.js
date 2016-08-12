import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { Link } from 'react-router'

class Classifier extends Component {
  render() {
    const { classifier } = this.props

    return (
      <div>
        <Link to={`/train/${classifier.id}`}>{classifier.title}</Link>
      </div>
    )
  }
}

Classifier.propTypes = {
  classifier: PropTypes.any.isRequired
}


export default connect(

)(Classifier)
