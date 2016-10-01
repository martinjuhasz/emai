import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { FormGroup, Radio, ControlLabel, Col, Panel, Clearfix, ButtonToolbar, Button, FormControl } from 'react-bootstrap/lib'
import { updateClassifier } from '../actions'

class ClassifierDetails extends Component {

  name(classifier) {
    switch (classifier.type) {
      case 3:
        return 'Logistic Regression Classifier'
      case 2:
        return 'Support Vector Machines'
      case 1:
        return 'Naive Bayes Classifier'
      default:
        return
    }
  }

  stop_words(classifier) {
    if(!classifier.settings || !classifier.settings.hasOwnProperty('stop_words')) {
      return
    }
    return 'Stop-Words filtered'
  }

  tfidf(classifier) {
    if(!classifier.settings || !classifier.settings.hasOwnProperty('idf')) {
      return
    }
    return 'IDF active'
  }

  ngram(classifier) {
    if(!classifier.settings || !classifier.settings.hasOwnProperty('ngram_range')) {
      return
    }
    return `N-Gram Range: 1-${classifier.settings.ngram_range}`
  }

  details(classifier) {
    return [
      this.name(classifier),
      this.stop_words(classifier),
      this.tfidf(classifier),
      this.ngram(classifier)
    ]
      .filter(detail => !!detail)
      .join(', ')
  }

  render() {
    const classifier = this.props.classifier
    if (!classifier) {
      return
    }

    return (
      <div>
        {this.details(classifier)}
      </div>
    )
  }
}

ClassifierDetails.propTypes = {
  classifier: PropTypes.any.isRequired
}

export default connect(

)(ClassifierDetails)
