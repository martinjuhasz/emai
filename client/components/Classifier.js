import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { Panel, Col, Glyphicon, ButtonGroup, Button } from 'react-bootstrap/lib'
import { LinkContainer } from 'react-router-bootstrap'

class Classifier extends Component {
  render() {
    const { classifier } = this.props

    return (
      <Panel>
        <Col xs={12} sm={12} md={12}>
          <h4>{classifier.title}</h4>
          <ButtonGroup>
            <LinkContainer to={`/train/${classifier.id}`}>
              <Button><Glyphicon glyph="cog"/> Details</Button>
            </LinkContainer>
          </ButtonGroup>
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
