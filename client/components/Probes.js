import React, { Component, PropTypes } from 'react'
import Paper from 'material-ui/Paper';
import { connect } from 'react-redux'
import {Table, TableBody, TableHeader, TableHeaderColumn, TableRow, TableRowColumn} from 'material-ui/Table';
import FlatButton from 'material-ui/FlatButton';
import RaisedButton from 'material-ui/RaisedButton';
import Dialog from 'material-ui/Dialog';

const style = {
  margin: 12,
};

class Probes extends Component {

  constructor() {
    super();

    this.state = {
    open_positive: false,
    open_negative: false
  };

  this.handleOpenPositive = () => {
    this.setState({open_positive: true});
  };

  this.handleOpenNegative = () => {
    this.setState({open_negative: true});
  };

  this.handleClose = () => {
    this.setState({open_negative: false, open_positive: false});
  };
  }


  render() {
    const { probes, title } = this.props

    const actions = [
      <FlatButton
        label="Cancel"
        primary={true}
        onTouchTap={this.handleClose} />
    ];

    return (
      <div>
        <h4>{title}</h4>
        <RaisedButton label="Show Positive Probes" primary={true} onTouchTap={this.handleOpenPositive} style={style} />
        <RaisedButton label="Show Negative Probes" primary={true} onTouchTap={this.handleOpenNegative} style={style} />
        <Dialog
          title="Positive Probes"
          actions={actions}
          modal={false}
          open={this.state.open_positive}
          onRequestClose={this.handleClose}
          autoScrollBodyContent={true}
        >
          <Table displaySelectAll={false}>
            <TableBody displayRowCheckbox={false}>
              {probes.positive.map(result => {
                return (<TableRow>
                  <TableRowColumn>{result[0]}</TableRowColumn>
                  <TableRowColumn>{result[1]}</TableRowColumn>
                </TableRow>)
                }
              )}
            </TableBody>
          </Table>
        </Dialog>

        <Dialog
          title="Negative Probes"
          actions={actions}
          modal={false}
          open={this.state.open_negative}
          onRequestClose={this.handleClose}
          autoScrollBodyContent={true}
        >
          <Table displaySelectAll={false}>
            <TableBody displayRowCheckbox={false}>
              {probes.negative.map(result => {
                return (<TableRow>
                  <TableRowColumn>{result[0]}</TableRowColumn>
                  <TableRowColumn>{result[1]}</TableRowColumn>
                </TableRow>)
                }
              )}
            </TableBody>
          </Table>
        </Dialog>
      </div>
    )
  }
}

Probes.propTypes = {
  probes: PropTypes.any.isRequired
}


export default connect(

)(Probes)
