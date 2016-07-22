import React, { Component, PropTypes } from 'react'

export default class MessageList extends Component {
  render() {
    return (
      <div>
        <h4>Messages</h4>
        <div>{this.props.children}</div>
      </div>
    )
  }
}

MessageList.propTypes = {
  children: PropTypes.node
}
