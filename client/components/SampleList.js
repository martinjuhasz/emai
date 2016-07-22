import React, { Component, PropTypes } from 'react'

export default class SampleList extends Component {
  render() {
    return (
      <div>
        <h3>Samples</h3>
        <div>{this.props.children}</div>
      </div>
    )
  }
}

SampleList.propTypes = {
  children: PropTypes.node
}
