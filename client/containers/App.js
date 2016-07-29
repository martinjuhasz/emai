import React, { Component } from 'react'
import SamplesContainer from './SamplesContainer'
import RecordingsContainer from './RecordingsContainer'
import RecordingContainer from './RecordingContainer'
import TrainingsContainer from './TrainingsContainer'
import TrainingContainer from './TrainingContainer'
import { Router, Route, Link, IndexRoute, hashHistory, browserHistory } from 'react-router'
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import Paper from 'material-ui/Paper';
import Menu from 'material-ui/Menu';
import MenuItem from 'material-ui/MenuItem';

export default class App extends Component {
  render() {
    return (
      <MuiThemeProvider>
        <Router history={hashHistory}>
          <Route path='/' component={Container}>
            <IndexRoute component={Home} />
            <Route path='recordings' component={RecordingsContainer}>
              <Route path=':recording_id' component={RecordingContainer} />
            </Route>

            <Route path='trainings' component={TrainingsContainer}>
              <Route path=':recording_id' component={TrainingContainer} />
            </Route>
            
            <Route path='*' component={NotFound} />
          </Route>
        </Router>
      </MuiThemeProvider>
    )
  }
}

const styles = {

  menu: {
    display: 'flex',
    flexDirection: 'column'
  },

  container: {
    display: 'flex',
    flexDirection: 'row wrap',
    width: '100%'
  },
  paperLeft: {
    flex: 1,
    height: '100%',
    margin: 10,
    padding: 10
  },
  paperRight: {
    flex: 4,
    margin: 10,
    padding: 10
  }
};

const Nav = () => (
  <Menu autoWidth={false} style={styles.menu}>
    <MenuItem containerElement={<Link to="/recordings" activeClassName="active" />} primaryText="Recordings" />
    <MenuItem containerElement={<Link to="/trainings" activeClassName="active" />} primaryText="Learn" />
    <MenuItem primaryText="Live" />
  </Menu>
);

const Container = (props) =>
  <div style={styles.container}>
    <Paper style={styles.paperLeft} zDepth={2}>
      <Nav />
    </Paper>
    <Paper style={styles.paperRight} zDepth={2}>
      {props.children}
    </Paper>
  </div>

const Home = () => <h1>This is Home</h1>

const NotFound = () => <h1>404.. This page is not found!</h1>