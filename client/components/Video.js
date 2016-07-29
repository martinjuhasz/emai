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
    const { video_id, title } = this.props
    const video_url = `videos/${video_id}.mp4`

    return (
      <video>
        <source src={video_url} type="video/mp4" />
      </video>
    )
  }
}

Classifier.propTypes = {
  video_id: PropTypes.string.isRequired
}


export default connect(

)(Classifier)
