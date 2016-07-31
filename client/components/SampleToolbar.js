import React, { Component, PropTypes } from 'react'
import {ButtonToolbar, ButtonGroup, Button, Glyphicon, DropdownButton, MenuItem } from 'react-bootstrap/lib'
import { LinkContainer } from 'react-router-bootstrap'

export default class SampleToolbar extends Component {
  render() {
  	const { recording_id, interval } = this.props
    return (
    	<ButtonToolbar>
    		<ButtonGroup>
          <DropdownButton title={`${interval}s`} id='sample_interval_dropdown'>
            <LinkContainer to={`/recordings/${recording_id}/samples/1`}>
              <MenuItem>1s</MenuItem>
            </LinkContainer>
            <LinkContainer to={`/recordings/${recording_id}/samples/3`}>
              <MenuItem>3s</MenuItem>
            </LinkContainer>
            <LinkContainer to={`/recordings/${recording_id}/samples/5`}>
              <MenuItem>5s</MenuItem>
            </LinkContainer>
            <LinkContainer to={`/recordings/${recording_id}/samples/10`}>
              <MenuItem>10s</MenuItem>
            </LinkContainer>
            <LinkContainer to={`/recordings/${recording_id}/samples/15`}>
              <MenuItem>15s</MenuItem>
            </LinkContainer>
          </DropdownButton>
          <Button onTouchTap={this.props.onReloadClicked}><Glyphicon glyph="repeat" /></Button>
    		</ButtonGroup>

        <ButtonGroup className='pull-right'>
          <Button onTouchTap={() => this.props.onUndoClicked()}><Glyphicon glyph="trash" /></Button>
          <Button onTouchTap={() => this.props.onSaveClicked()}><Glyphicon glyph="ok" /></Button>
        </ButtonGroup>

    		<ButtonGroup className='pull-right'>
    		  <Button bsStyle="danger" onTouchTap={() => this.props.onClassifyClicked(2)}><Glyphicon glyph="thumbs-down" /></Button>
          <Button bsStyle="warning" onTouchTap={() => this.props.onClassifyClicked(1)}><Glyphicon glyph="minus" /></Button>
          <Button bsStyle="success" onTouchTap={() => this.props.onClassifyClicked(3)}><Glyphicon glyph="thumbs-up" /></Button>
    		</ButtonGroup>

    	</ButtonToolbar>
    )
  }
}

SampleToolbar.propTypes = {
  recording_id: PropTypes.string.isRequired,
  interval: PropTypes.string.isRequired,
  onReloadClicked: PropTypes.func.isRequired,
  onClassifyClicked: PropTypes.func.isRequired,
  onUndoClicked: PropTypes.func.isRequired,
  onSaveClicked: PropTypes.func.isRequired
}
