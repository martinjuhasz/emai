import React, { Component, PropTypes } from 'react'
import MessageList from './MessageList'
import Message from './Message'
import Paper from 'material-ui/Paper';
import FloatingActionButton from 'material-ui/FloatingActionButton';
import ContentAdd from 'material-ui/svg-icons/content/add';
import ContentRemove from 'material-ui/svg-icons/content/remove';
import ContentClear from 'material-ui/svg-icons/content/clear';
import {red500, grey500, green500} from 'material-ui/styles/colors';

var styles = {
  paper: {
    padding: 10,
    margin: 10,
    maxWidth: 300,
  },
  button: {
    margin: 10
  }
};

export default class Sample extends Component {
  render() {
    const { sample } = this.props

    return (
      <Paper style={styles.paper}>
        {sample.messages.map(message =>
            <Message
              key={message.identifier}
              message={message} />
          )}
        <FloatingActionButton mini={true} onClick={() => this.props.onClassifySampleClicked(2)} style={styles.button} backgroundColor={red500}>
          <ContentRemove />
        </FloatingActionButton>
        <FloatingActionButton mini={true} onClick={() => this.props.onClassifySampleClicked(1)} style={styles.button} backgroundColor={grey500}>
          <ContentClear />
        </FloatingActionButton>
        <FloatingActionButton mini={true} onClick={() => this.props.onClassifySampleClicked(3)} style={styles.button} backgroundColor={green500}>
          <ContentAdd />
        </FloatingActionButton>
      </Paper>
    )
  }
}

Sample.propTypes = {
  sample: PropTypes.shape({
    message: PropTypes.any
  }).isRequired,
  onClassifySampleClicked: PropTypes.func.isRequired
}
