import React, { Component, PropTypes } from 'react'
import {ButtonToolbar, ButtonGroup, Button, Glyphicon } from 'react-bootstrap/lib'

export default class VideoToolbar extends Component {

  render() {
    return (
      <ButtonToolbar>
        <ButtonGroup>
          <Button onTouchTap={() => this.props.onPlayClicked()}><Glyphicon glyph="play"/></Button>
          <Button onTouchTap={() => this.props.onStopClicked()}><Glyphicon glyph="stop"/></Button>
        </ButtonGroup>

      </ButtonToolbar>
    )
  }
}

VideoToolbar.propTypes = {
  onPlayClicked: PropTypes.func.isRequired,
  onStopClicked: PropTypes.func.isRequired
}
