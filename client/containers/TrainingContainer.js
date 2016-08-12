import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Classifier from '../components/Classifier'
import { getClassifiers } from '../actions'

const classifiers_list = (classifiers) => {
  return (
      <div>
        {classifiers.map(classifier =>
          <Classifier
            key={classifier.id}
            classifier={classifier} />
        )}
      </div>
    )
}

class TrainingContainer extends Component {

  componentDidMount() {
    this.props.getClassifiers()
  }

  render() {
    const { classifiers } = this.props
    return (
      <div> {this.props.children || classifiers_list(classifiers)} </div>
    )
  }
}

TrainingContainer.propTypes = {
  classifiers: PropTypes.any.isRequired,
  children: PropTypes.node,
  getClassifiers: PropTypes.func.isRequired
}

function mapStateToProps(state) {
  return {
    classifiers: state.classifiers.all
  }
}

export default connect(
  mapStateToProps,
  {
    getClassifiers
  }
)(TrainingContainer)
