import React, { Component, PropTypes } from 'react'

const styles = {
    samples: {
      display: 'flex',
      flexWrap: 'wrap',
    },
  };

export default class SampleList extends Component {
  render() {
    return (
      <div>
        <h3>{this.props.children && 'Samples'}</h3>
        <div style={styles.samples}>{this.props.children && this.props.children}</div>
      </div>
    )
  }
}

SampleList.propTypes = {
  children: PropTypes.node
}
