import React, { Component, PropTypes } from 'react'

export default class MessageList extends Component {
  render() {
    return (
      <div>
        <div>{this.props.children}</div>
      </div>
    )
  }
}

MessageList.propTypes = {
  children: PropTypes.node
}
