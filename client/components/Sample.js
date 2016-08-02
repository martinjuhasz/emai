import React, { Component, PropTypes } from 'react'
import { getMessages } from '../reducers/samples'
import { connect } from 'react-redux'
import {ListGroup } from 'react-bootstrap/lib'
import MessageGroupItem from './MessageGroupItem'

class Sample extends Component {
  render() {
    const { messages, selected_message } = this.props
    return (
      <ListGroup>
          {messages.map(message => {
            return (
              <MessageGroupItem
                onTouchTap={() => this.props.onMessageClicked(message._id)}
                key={message._id}
                message={message}
                selected_message={selected_message} />
            )
          })}
      </ListGroup>
    )
  }
}

Sample.propTypes = {
  messages: PropTypes.array.isRequired,
  selected_message: PropTypes.string,
  onMessageClicked: PropTypes.func.isRequired
}


function mapStateToProps(state, ownProps) {
  return {
    messages: getMessages(state, ownProps.sample.messages)
  }
}

export default connect(
  mapStateToProps
)(Sample)
