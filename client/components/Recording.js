import React, { Component, PropTypes } from 'react'
import { Panel, Col, Image, ButtonGroup, Button, Glyphicon, ButtonToolbar } from 'react-bootstrap/lib'
import { LinkContainer } from 'react-router-bootstrap'

export default class Recording extends Component {
  render() {
    const { recording, path } = this.props
    const logo_url = recording.logo || 'https://unsplash.it/300?random'
    return (
      <Panel>
        <Col xs={2} sm={2} md={2}>
          <Image src={logo_url} rounded responsive />
        </Col>
        <Col xs={10} sm={10} md={10}>
          <h4>{recording.display_name} <small>{recording.started}</small></h4>
          <ButtonToolbar>
            <ButtonGroup>
              <LinkContainer to={`/recordings/${recording.id}/samples/10`}>
                <Button>Show Samples</Button>
              </LinkContainer>
            </ButtonGroup>
            <ButtonGroup>
              <LinkContainer to={`/recordings/${recording.id}/replay`}>
                <Button><Glyphicon glyph="play"/></Button>
              </LinkContainer>
            </ButtonGroup>
          </ButtonToolbar>
        </Col>
      </Panel>
    )
  }
}

Recording.propTypes = {
  recording: PropTypes.shape({
    id: PropTypes.string.isRequired,
    display_name: PropTypes.string.isRequired,
    started: PropTypes.string.isRequired,
    stopped: PropTypes.string.isRequired
  }),
  path: PropTypes.string.isRequired
}
