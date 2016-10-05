import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Recording from '../components/Recording'
import { getRecordings } from '../actions/recordings'
import { Row, Col, FormControl, Button, Glyphicon } from 'react-bootstrap/lib'

const recording_list = (recordings) => {
  return (
      <div>
        <h2>Recordings</h2>
        {recording_search()}
        <div className="hspace">
          {recordings.map(recording =>
            <Recording
              key={recording.id}
              recording={recording}
              path='recordings' />
          )}
        </div>
      </div>
    )
}

const recording_search = () => {
  return (
    <Row>
      <Col xs={5} sm={5} md={5}>
        <FormControl type="text" placeholder="Username" />
      </Col>
      <Col xs={3} sm={3} md={3}>
        <Button type="submit">
          <Glyphicon glyph="record"/> Record
        </Button>
      </Col>
      <Col xs={4} sm={4} md={4}>

      </Col>
    </Row>
  )
}

class RecordingsContainer extends Component {

  componentDidMount() {
    this.props.getRecordings()
  }

  render() {
    const { recordings } = this.props
    return (
      <div> {this.props.children || recording_list(recordings)} </div>
    )
  }
}

RecordingsContainer.propTypes = {
  recordings: PropTypes.any.isRequired,
  children: PropTypes.node,
  getRecordings: PropTypes.func.isRequired
}

function mapStateToProps(state) {
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps,
  { getRecordings }
)(RecordingsContainer)
