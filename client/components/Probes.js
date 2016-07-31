import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'

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

    return (
      <div>
        <h4>{title}</h4>
        <div onTouchTap={this.handleOpenPositive}>Show Positive Probes</div>
        <div onTouchTap={this.handleOpenNegative}>Show Negative Probes</div>

        <div>
          {probes.positive.map(result => {
            return (<TableRow>
              <TableRowColumn>{result[0]}</TableRowColumn>
              <TableRowColumn>{result[1]}</TableRowColumn>
            </TableRow>)
            }
          )}
        </div>

        <div>
          {probes.negative.map(result => {
            return (<TableRow>
              <TableRowColumn>{result[0]}</TableRowColumn>
              <TableRowColumn>{result[1]}</TableRowColumn>
            </TableRow>)
            }
          )}
        </div>
      </div>
    )
  }
}

Probes.propTypes = {
  probes: PropTypes.any.isRequired
}


export default connect(

)(Probes)
