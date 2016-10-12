import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import Classifier from '../components/Classifier'
import { getClassifiers, deleteClassifier, createClassifier } from '../actions'
import { Row, Col, FormControl, Button, Glyphicon } from 'react-bootstrap/lib'

class ClassifiersContainer extends Component {

  constructor() {
    super()
    this.state = {
      add_classifier: null,
    }
    this.renderClassifiers = this.renderClassifiers.bind(this)
    this.renderNewClassifierInput = this.renderNewClassifierInput.bind(this)
    this.onCreateClassifierClicked = this.onCreateClassifierClicked.bind(this)
    this.classifierInputChanged = this.classifierInputChanged.bind(this)
  }

  componentDidMount() {
    this.props.getClassifiers()
  }

  classifierInputChanged(event) {
    this.setState({add_classifier: event.target.value})
  }

  onCreateClassifierClicked() {
    if(this.state.add_classifier) {
      this.props.createClassifier(this.state.add_classifier)
    }
  }

  renderNewClassifierInput() {
    return (
      <Row>
        <Col xs={5} sm={5} md={5}>
          <FormControl type="text" placeholder="Title" onChange={this.classifierInputChanged} />
        </Col>
        <Col xs={3} sm={3} md={3}>
          <Button onTouchTap={this.onCreateClassifierClicked}>
            <Glyphicon glyph="plus"/> Add
          </Button>
        </Col>
        <Col xs={4} sm={4} md={4}>

        </Col>
      </Row>
    )
  }

  renderClassifiers(classifiers) {
    return (
      <div>
        <h2>Classifiers</h2>
        {this.renderNewClassifierInput()}
        <div className="hspace">
          {classifiers.map(classifier =>
            <Classifier
              key={classifier.id}
              onDeleteClicked={() => this.props.deleteClassifier(classifier.id)}
              classifier={classifier} />
          )}
        </div>
      </div>
    )
  }

  render() {
    const { classifiers } = this.props
    return (
      <div> {this.props.children || this.renderClassifiers(classifiers)} </div>
    )
  }
}

ClassifiersContainer.propTypes = {
  classifiers: PropTypes.any.isRequired,
  children: PropTypes.node,
  getClassifiers: PropTypes.func.isRequired,
  deleteClassifier: PropTypes.func.isRequired,
  createClassifier: PropTypes.func.isRequired
}

function mapStateToProps(state) {
  return {
    classifiers: state.classifiers.all
  }
}

export default connect(
  mapStateToProps,
  {
    getClassifiers,
    deleteClassifier,
    createClassifier
  }
)(ClassifiersContainer)
