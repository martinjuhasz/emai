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
            const message_id = message._id || message.id
            return (
              <MessageGroupItem
                onTouchTap={() => this.props.onMessageClicked(message_id)}
                key={message_id}
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

  }
}

export default connect(
  mapStateToProps
)(Sample)
