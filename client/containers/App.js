import React, { Component, PropTypes } from 'react'
import SamplesContainer from './SamplesContainer'
import RecordingsContainer from './RecordingsContainer'
import RecordingContainer from './RecordingContainer'
import TrainingsContainer from './TrainingsContainer'
import TrainingContainer from './TrainingContainer'
import { Router, Route,  IndexRoute, hashHistory } from 'react-router'
import {Grid, Row, Col, Nav, NavItem } from 'react-bootstrap/lib'
import { LinkContainer } from 'react-router-bootstrap'

export default class App extends Component {
  render() {
    return (
      <Router history={hashHistory}>
        <Route path='/' component={Container}>
          <IndexRoute component={Home} />
          <Route path='recordings' component={RecordingsContainer}>
            <Route path=':recording_id' component={RecordingContainer}>
              <Route path='samples/:interval' component={SamplesContainer} />
            </Route>
          </Route>

          <Route path='train' component={TrainingsContainer}>
            <Route path=':recording_id' component={TrainingContainer} />
          </Route>

          <Route path='*' component={NotFound} />
        </Route>
      </Router>
    )
  }
}

class Navigation extends Component {
  render() {
    return(
      <Nav bsStyle='pills'>
        <LinkContainer to='recordings'>
          <NavItem>Recordings</NavItem>
        </LinkContainer>
        <LinkContainer to='train'>
          <NavItem>Train</NavItem>
        </LinkContainer>
        <LinkContainer to='live'>
          <NavItem>Live</NavItem>
        </LinkContainer>
      </Nav>
    )
  }
}

class Container extends Component {
  render() {
    return(
      <Grid>
        <Row>
          <Col xs={12} sm={12} className='hspace'><Navigation /></Col>
          <Col xs={12} sm={12}>{this.props.children}</Col>
        </Row>
      </Grid>
    )
  }
}
Container.propTypes = {
  children: PropTypes.node.isRequired
}


class Home extends Component {
  render() {
    return(
      <h1>This is Home</h1>
    )
  }
}

class NotFound extends Component {
  render() {
    return(
      <h1>404.. This page is not found!</h1>
    )
  }
}
