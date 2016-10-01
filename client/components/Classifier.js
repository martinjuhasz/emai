import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { Panel, Col, Glyphicon, ButtonGroup, Button, ButtonToolbar } from 'react-bootstrap/lib'
import { LinkContainer } from 'react-router-bootstrap'

class Classifier extends Component {
  render() {
    const { classifier } = this.props

    return (
      <Panel>
        <Col xs={12} sm={12} md={12}>
          <h4>{classifier.title}</h4>
          <ButtonToolbar>
            <ButtonGroup>
              <LinkContainer to={`/train/${classifier.id}`}>
                <Button><Glyphicon glyph="cog"/> Details</Button>
              </LinkContainer>
            </ButtonGroup>
            <ButtonGroup>
              <LinkContainer to={`/train/${classifier.id}/train`}>
                <Button><Glyphicon glyph="education"/> Train</Button>
              </LinkContainer>
            </ButtonGroup>
          </ButtonToolbar>
        </Col>
      </Panel>
    )
  }
}

Classifier.propTypes = {
  classifier: PropTypes.any.isRequired
}


export default connect(

)(Classifier)
