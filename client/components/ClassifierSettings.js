import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { FormGroup, Radio, ControlLabel, Col, Panel, Clearfix, ButtonToolbar, Button, Row } from 'react-bootstrap/lib'
import { updateClassifier } from '../actions'
import Select from 'react-select';

class ClassifierSettings extends Component {

  constructor() {
    super()
    this.state = {
      selected_type: null,
      selected_ngram: null,
      selected_stopwords: null,
      selected_idf: null,
      selected_c: null,
      selected_alpha: null,
      selected_gamma: null,
      selected_recordings: null
    }
    this.onSaveClicked = this.onSaveClicked.bind(this)
    this.onResetClicked = this.onResetClicked.bind(this)
    this.onValueChange = this.onValueChange.bind(this)
    this.recordingOptions = this.recordingOptions.bind(this)
    this.setStateFromClassifier = this.setStateFromClassifier.bind(this)
    this.onChangeRecordingSelection = this.onChangeRecordingSelection.bind(this)
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
    if (classifier.settings && classifier.settings.hasOwnProperty('c')) {
      this.setState({selected_c: classifier.settings.c.toString()})
    }
    if (classifier.settings && classifier.settings.hasOwnProperty('gamma')) {
      this.setState({selected_gamma: classifier.settings.gamma.toString()})
    }
    if (classifier.settings && classifier.settings.hasOwnProperty('alpha')) {
      this.setState({selected_alpha: classifier.settings.alpha.toString()})
    }

    const recordings = this.props.recordings
      .filter(recording => classifier.training_sets.includes(recording.id))
      .map(recording => {
        return { value: recording.id, label: `${recording.display_name} - ${recording.id}` }
      })
    this.setState({selected_recordings: recordings})
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
    if(this.state.selected_c) {
      settings.c = parseInt(this.state.selected_c)
    }
    if(this.state.selected_gamma) {
      settings.gamma = parseInt(this.state.selected_gamma)
    }
    if(this.state.selected_alpha) {
      settings.alpha = parseInt(this.state.selected_alpha)
    }
    const training_sets = this.state.selected_recordings.map(selection => selection.value)
    const cls_type = (this.state.selected_type === null) ? null : parseInt(this.state.selected_type)
    this.props.updateClassifier(classifier.id, cls_type, settings, training_sets)
  }

  onResetClicked() {
    if(!this.state.selected_type) {
      this.setState({selected_type: '1'})
    }

    this.setState({selected_ngram: '1'})
    this.setState({selected_stopwords: 'true'})
    this.setState({selected_idf: 'true'})
    this.setState({selected_c: '4'})
    this.setState({selected_alpha: '2'})
    this.setState({selected_gamma: '5'})
  }

  onChangeRecordingSelection(value) {
    this.setState({selected_recordings: value})
  }

  recordingOptions() {
    return this.props.recordings.map(recording => { return { value: recording.id, label: `${recording.display_name} - ${recording.id}` }})
  }

  render() {
    return (
      <div>
        <h3>Settings</h3>

        <Row>
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
            <Select
              multi={true}
              name="form-field-name"
              value={this.state.selected_recordings}
              options={this.recordingOptions()}
              onChange={this.onChangeRecordingSelection}
            />
          </Col>
        </Row>
        <Row>
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
          <Col xs={12} sm={6} md={6}>
            {(this.state.selected_type === '3' || this.state.selected_type === '2') &&
            <Panel>
              <FormGroup>
                <Col xs={12} sm={2} md={2} componentClass={ControlLabel}>
                  C
                </Col>
                <Col xs={12} sm={10} md={10}>
                  <Radio inline checked={this.state.selected_c === '1'} onChange={() => {this.onValueChange('selected_c', '1')}}>0.25</Radio>
                  <Radio inline checked={this.state.selected_c === '2'} onChange={() => {this.onValueChange('selected_c', '2')}}>0.5</Radio>
                  <Radio inline checked={this.state.selected_c === '3'} onChange={() => {this.onValueChange('selected_c', '3')}}>1</Radio>
                  <Radio inline checked={this.state.selected_c === '4'} onChange={() => {this.onValueChange('selected_c', '4')}}>2</Radio>
                  <Radio inline checked={this.state.selected_c === '5'} onChange={() => {this.onValueChange('selected_c', '5')}}>4</Radio>
                </Col>
                <Clearfix />
              </FormGroup>
            </Panel>
            }
            {this.state.selected_type === '2' &&
            <Panel>
              <FormGroup>
                <Col xs={12} sm={2} md={2} componentClass={ControlLabel}>
                  Gamma
                </Col>
                <Col xs={12} sm={10} md={10}>
                  <Radio inline checked={this.state.selected_gamma === '1'} onChange={() => {this.onValueChange('selected_gamma', '1')}}>auto</Radio>
                  <Radio inline checked={this.state.selected_gamma === '2'} onChange={() => {this.onValueChange('selected_gamma', '2')}}>0.01</Radio>
                  <Radio inline checked={this.state.selected_gamma === '3'} onChange={() => {this.onValueChange('selected_gamma', '3')}}>0.1</Radio>
                  <Radio inline checked={this.state.selected_gamma === '4'} onChange={() => {this.onValueChange('selected_gamma', '4')}}>0.25</Radio>
                  <Radio inline checked={this.state.selected_gamma === '5'} onChange={() => {this.onValueChange('selected_gamma', '5')}}>0.5</Radio>
                  <Radio inline checked={this.state.selected_gamma === '6'} onChange={() => {this.onValueChange('selected_gamma', '6')}}>0.75</Radio>
                </Col>
                <Clearfix />
              </FormGroup>
            </Panel>
            }
            {this.state.selected_type === '1' &&
            <Panel>
              <FormGroup>
                <Col xs={12} sm={2} md={2} componentClass={ControlLabel}>
                  Alpha
                </Col>
                <Col xs={12} sm={10} md={10}>
                  <Radio inline checked={this.state.selected_alpha === '1'} onChange={() => {this.onValueChange('selected_alpha', '1')}}>0.25</Radio>
                  <Radio inline checked={this.state.selected_alpha === '2'} onChange={() => {this.onValueChange('selected_alpha', '2')}}>0.5</Radio>
                  <Radio inline checked={this.state.selected_alpha === '3'} onChange={() => {this.onValueChange('selected_alpha', '3')}}>1</Radio>
                  <Radio inline checked={this.state.selected_alpha === '4'} onChange={() => {this.onValueChange('selected_alpha', '4')}}>2</Radio>
                  <Radio inline checked={this.state.selected_alpha === '5'} onChange={() => {this.onValueChange('selected_alpha', '5')}}>4</Radio>
                </Col>
                <Clearfix />
              </FormGroup>
            </Panel>
            }
          </Col>
        </Row>
        <Row>
          <ButtonToolbar>
            <Button bsStyle="danger" onTouchTap={() => {this.onSaveClicked()}}>Update</Button>
            <Button onTouchTap={() => {this.onResetClicked()}}>Reset to best settings</Button>
          </ButtonToolbar>
        </Row>
      </div>
    )
  }
}

ClassifierSettings.propTypes = {
  classifier: PropTypes.any.isRequired,
  updateClassifier: PropTypes.func.isRequired,
  recordings: PropTypes.any.isRequired
}


function mapStateToProps(state) {
  return {
    recordings: state.recordings.all
  }
}

export default connect(
  mapStateToProps,
  {
    updateClassifier
  }
)(ClassifierSettings)
