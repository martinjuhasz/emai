import React, { Component, PropTypes } from 'react'
import Paper from 'material-ui/Paper';
import { connect } from 'react-redux'
import {Table, TableBody, TableHeader, TableHeaderColumn, TableRow, TableRowColumn} from 'material-ui/Table';


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
