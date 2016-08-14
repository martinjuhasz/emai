import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { FormGroup, Radio, ControlLabel, Col, Panel, Clearfix, ButtonToolbar, Button } from 'react-bootstrap/lib'
import { updateClassifier } from '../actions'

class ClassifierSettings extends Component {

  constructor() {
    super()
    this.state = {
      selected_type: null,
      selected_ngram: null,
      selected_stopwords: null,
      selected_idf: null
    }
    this.onSaveClicked = this.onSaveClicked.bind(this)
    this.onTestClicked = this.onTestClicked.bind(this)
    this.onValueChange = this.onValueChange.bind(this)
    this.setStateFromClassifier = this.setStateFromClassifier.bind(this)
  }

  componentDidMount() {
    this.setStateFromClassifier()
  }

  setStateFromClassifier() {
    if (!this.props.classifier) {
      return
    }
    const classifier = this.props.classifier
    if (classifier.type) {
      this.setState({selected_type: classifier.type.toString()})
    }
    if (classifier.settings && classifier.settings.ngram_range) {
      this.setState({selected_ngram: classifier.settings.ngram_range.toString()})
    }
    if (classifier.settings && classifier.settings.hasOwnProperty('stop_words')) {
      this.setState({selected_stopwords: classifier.settings.stop_words.toString()})
    }
    if (classifier.settings && classifier.settings.hasOwnProperty('idf')) {
      this.setState({selected_idf: classifier.settings.idf.toString()})
    }
  }

  onValueChange(property, value) {
    this.setState({[property]: value})
  }

  onSaveClicked() {
    const { classifier } = this.props
    if(!classifier) {
      return
    }
    const settings = {}
    if(this.state.selected_ngram) {
      settings.ngram_range = parseInt(this.state.selected_ngram)
    }
    if(this.state.selected_stopwords !== null) {
      settings.stop_words = this.state.selected_stopwords === 'true'
    }
    if(this.state.selected_idf !== null) {
      settings.idf = this.state.selected_idf === 'true'
    }
    const cls_type = (this.state.selected_type === null) ? null : parseInt(this.state.selected_type)
    this.props.updateClassifier(classifier.id, settings, cls_type, null)
  }

  onTestClicked() {

  }

  render() {
    const { classifier } = this.props

    return (
      <div>
        <h3>Settings</h3>

        <Col xs={12} sm={6} md={6}>
          <Panel>
            <FormGroup>
              <ControlLabel>Classifier Type</ControlLabel>
              <Radio checked={this.state.selected_type === '3'} onChange={() => {this.onValueChange('selected_type', '3')}}>Logistic Regression</Radio>
              <Radio checked={this.state.selected_type === '2'} onChange={() => {this.onValueChange('selected_type', '2')}}>Support Vector Machine</Radio>
              <Radio checked={this.state.selected_type === '1'} onChange={() => {this.onValueChange('selected_type', '1')}}>Naive Bayes</Radio>
            </FormGroup>
          </Panel>
        </Col>

        <Col xs={12} sm={6} md={6}>
          <Panel>
            <FormGroup>
              <Col xs={12} sm={4} md={4} componentClass={ControlLabel}>
                N-Gram Range
              </Col>
              <Col xs={12} sm={8} md={8}>
                <Radio inline checked={this.state.selected_ngram === '1'} onChange={() => {this.onValueChange('selected_ngram', '1')}}>1-1</Radio>
                <Radio inline checked={this.state.selected_ngram === '2'} onChange={() => {this.onValueChange('selected_ngram', '2')}}>1-2</Radio>
                <Radio inline checked={this.state.selected_ngram === '3'} onChange={() => {this.onValueChange('selected_ngram', '3')}}>1-3</Radio>
              </Col>
              <Clearfix />
            </FormGroup>
            <FormGroup>
              <Col xs={12} sm={4} md={4} componentClass={ControlLabel}>
                Stop Words
              </Col>
              <Col xs={12} sm={8} md={8}>
                <Radio inline checked={this.state.selected_stopwords === 'true'} onChange={() => {this.onValueChange('selected_stopwords', 'true')}}>Yes</Radio>
                <Radio inline checked={this.state.selected_stopwords === 'false'} onChange={() => {this.onValueChange('selected_stopwords', 'false')}}>No</Radio>
              </Col>
              <Clearfix />
            </FormGroup>
            <FormGroup>
              <Col xs={12} sm={4} md={4} componentClass={ControlLabel}>
                IDF
              </Col>
              <Col xs={12} sm={8} md={8}>
                <Radio inline checked={this.state.selected_idf === 'true'} onChange={() => {this.onValueChange('selected_idf', 'true')}}>Yes</Radio>
                <Radio inline checked={this.state.selected_idf === 'false'} onChange={() => {this.onValueChange('selected_idf', 'false')}}>No</Radio>
              </Col>
              <Clearfix />
            </FormGroup>
          </Panel>
        </Col>

        <ButtonToolbar>
          <Button bsStyle="danger" onTouchTap={() => {this.onSaveClicked()}}>Update</Button>
          <Button onTouchTap={() => {this.onTestClicked()}}>Test for best settings</Button>
        </ButtonToolbar>
      </div>
    )
  }
}

ClassifierSettings.propTypes = {
  classifier: PropTypes.any.isRequired,
  updateClassifier: PropTypes.func.isRequired
}


function mapStateToProps(state, ownProps) {
  return {
  }
}

export default connect(
  mapStateToProps,
  {
    updateClassifier
  }
)(ClassifierSettings)
