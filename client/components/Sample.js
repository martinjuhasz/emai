import React, { Component, PropTypes } from 'react'
import MessageList from './MessageList'
import Message from './Message'
import Paper from 'material-ui/Paper';
import FloatingActionButton from 'material-ui/FloatingActionButton';
import ContentAdd from 'material-ui/svg-icons/content/add';
import ContentRemove from 'material-ui/svg-icons/content/remove';
import ContentClear from 'material-ui/svg-icons/content/clear';
import {red500, grey500, green500} from 'material-ui/styles/colors';
import { checkMessage } from '../actions'
import { getMessages } from '../reducers/samples'
import { connect } from 'react-redux'

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

class Sample extends Component {
  render() {
    const { messages } = this.props

    return (
      <Paper style={styles.paper}>
        {messages.map(message =>
            <Message
              key={message._id}
              onClick={() => this.props.onCheckMessageClicked(message._id)}
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
  sample: PropTypes.any.isRequired,
  onClassifySampleClicked: PropTypes.func.isRequired,
  onCheckMessageClicked: PropTypes.func.isRequired
}


function mapStateToProps(state, ownProps) { 
  return {
    messages: getMessages(state, ownProps.sample.messages)
  }
}

export default connect(
  mapStateToProps
)(Sample)
