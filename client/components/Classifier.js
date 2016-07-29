import React, { Component, PropTypes } from 'react'
import Paper from 'material-ui/Paper';
import { connect } from 'react-redux'
import {Table, TableBody, TableHeader, TableHeaderColumn, TableRow, TableRowColumn} from 'material-ui/Table';
import ClassifierResult from './ClassifierResult';
import SettingsResult from './SettingsResult';
import Probes from './Probes';

var styles = {
  paper: {
    padding: 10,
    margin: 10,
    maxWidth: 500,
  },
  button: {
    margin: 10
  }
};

class Classifier extends Component {
  render() {
    const { classifier, title } = this.props

    return (
      <Paper style={styles.paper} zDepth={2}>
        <h3>{title}</h3>
        <SettingsResult title="Settings" result={classifier.settings} />
        <ClassifierResult title="Positive" result={classifier.results.positive} />
        <ClassifierResult title="Negative" result={classifier.results.negative} />
        <Probes title="Probes" probes={classifier.probes} />
      </Paper>
    )
  }
}

Classifier.propTypes = {
  classifier: PropTypes.any.isRequired
}


export default connect(

)(Classifier)
