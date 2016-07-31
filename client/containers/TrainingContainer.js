import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byRecording } from '../reducers/classifiers'
import { Link } from 'react-router'
import { getClassifiers } from '../actions'
import Classifier from '../components/Classifier'

const styles = {
    classifiers: {
      display: 'flex',
      flexWrap: 'wrap',
    },
  };

class TrainingContainer extends Component {

  render() {
    const { classifiers } = this.props

    return (
      <div>
        <h2>Classifiers</h2>
        <div onTouchTap={() => this.props.onReloadClicked()}>Train Classifiers</div>

        <div style={styles.classifiers}>
          {classifiers ? <Classifier title="Logistic Regression" classifier={classifiers['logreg']} /> : null }
          {classifiers ? <Classifier title="Support Vector Machines" classifier={classifiers['svm']} /> : null }
          {classifiers ? <Classifier title="Multinomial Naive Bayes" classifier={classifiers['nb']} /> : null }
        </div>
      </div>
    )
  }
}

TrainingContainer.propTypes = {
  classifiers: PropTypes.shape({
    svm: PropTypes.any.isRequired,
    logreg: PropTypes.any.isRequired,
    nb: PropTypes.any.isRequired
  }),
  onReloadClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) { 
  return {
    classifiers: byRecording(state, ownProps.params.recording_id)
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    onReloadClicked: () => {
      dispatch(getClassifiers(ownProps.params.recording_id))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingContainer)
