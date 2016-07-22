import React, { Component, PropTypes } from 'react'
import MessageList from './MessageList'
import Message from './Message'

export default class Sample extends Component {
  render() {
    const { sample } = this.props

    return (
      <div>
        <MessageList title="Samples">
          {sample.messages.map(message =>
            <Message
              key={message.identifier}
              message={message} />
          )}
        </MessageList>
        <button type="button" onClick={() => this.props.onClassifySampleClicked(2)}>Negative</button>
        <button type="button" onClick={() => this.props.onClassifySampleClicked(1)}>Invalid</button>
        <button type="button" onClick={() => this.props.onClassifySampleClicked(3)}>Positive</button>
      </div>
    )
  }
}

Sample.propTypes = {
  sample: PropTypes.shape({
    message: PropTypes.any
  }).isRequired,
  onClassifySampleClicked: PropTypes.func.isRequired
}
