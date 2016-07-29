import React, { Component, PropTypes } from 'react'
import Paper from 'material-ui/Paper';
import { connect } from 'react-redux'
import {Table, TableBody, TableHeader, TableHeaderColumn, TableRow, TableRowColumn} from 'material-ui/Table';


class ClassifierResult extends Component {
  render() {
    const { result, title } = this.props

    return (
      <div>
        <h4>{title}</h4>
          <Table displaySelectAll={false}>
            <TableBody displayRowCheckbox={false}>
              <TableRow>
                <TableRowColumn>precision</TableRowColumn>
                <TableRowColumn>{result.precision}</TableRowColumn>
              </TableRow>
              <TableRow>
                <TableRowColumn>fscore</TableRowColumn>
                <TableRowColumn>{result.fscore}</TableRowColumn>
              </TableRow>
              <TableRow>
                <TableRowColumn>recall</TableRowColumn>
                <TableRowColumn>{result.recall}</TableRowColumn>
              </TableRow>
              <TableRow>
                <TableRowColumn>support</TableRowColumn>
                <TableRowColumn>{result.support}</TableRowColumn>
              </TableRow>
            </TableBody>
          </Table>
      </div>
    )
  }
}

ClassifierResult.propTypes = {
  result: PropTypes.any.isRequired
}


export default connect(

)(ClassifierResult)
