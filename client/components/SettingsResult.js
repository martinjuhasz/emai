import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'


class SettingsResult extends Component {
  render() {
    const { result, title } = this.props
    const result_txt = JSON.stringify(result)

    return (
      <div>
        <h4>{title}</h4>
        {result_txt}
      </div>
    )
  }
}

SettingsResult.propTypes = {
  result: PropTypes.any.isRequired
}


export default connect(

)(SettingsResult)
