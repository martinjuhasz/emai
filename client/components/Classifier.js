import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import ClassifierResult from './ClassifierResult';
import SettingsResult from './SettingsResult';
import Probes from './Probes';


class Classifier extends Component {
  render() {
    const { classifier, title } = this.props

    return (
      <div>
        <h3>{title}</h3>
        <SettingsResult title="Settings" result={classifier.settings} />
        <ClassifierResult title="Positive" result={classifier.results.positive} />
        <ClassifierResult title="Negative" result={classifier.results.negative} />
        <Probes title="Probes" probes={classifier.probes} />
      </div>
    )
  }
}

Classifier.propTypes = {
  classifier: PropTypes.any.isRequired
}


export default connect(

)(Classifier)
