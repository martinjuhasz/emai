import React, { Component, PropTypes } from 'react'
import SamplesContainer from './SamplesContainer'
import RecordingsContainer from './RecordingsContainer'
import RecordingContainer from './RecordingContainer'
import TrainingsContainer from './TrainingsContainer'
import TrainingContainer from './TrainingContainer'
import { Router, Route, Link, IndexRoute, hashHistory } from 'react-router'
import {Grid, Row, Col } from 'react-bootstrap/lib'

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

          <Route path='trainings' component={TrainingsContainer}>
            <Route path=':recording_id' component={TrainingContainer} />
          </Route>

          <Route path='*' component={NotFound} />
        </Route>
      </Router>
    )
  }
}

class Nav extends Component {
  render() {
    return(
      <div>
        <Link to="/recordings" activeClassName="active">Recordings</Link>
        <Link to="/trainings" activeClassName="active">Learn</Link>
        <Link to="/live" activeClassName="active">Live</Link>
      </div>
    )
  }
}

class Container extends Component {
  render() {
    return(
      <Grid>
        <Row>
          <Col xs={12} sm={12}><Nav /></Col>
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
