import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { byId as classifierById } from '../reducers/classifiers'
import { trainClassifier, learnClassifier } from '../actions'
import ClassifierDetails from '../components/ClassifierDetails'
import ClassifierPerformance from '../components/ClassifierPerformance'
import { ButtonToolbar, Button, Col, Row, Panel, InputGroup, FormControl, ButtonGroup, DropdownButton, MenuItem  } from 'react-bootstrap/lib'
import Review from '../components/Review'

class TrainingContainer extends Component {

  constructor() {
    super()
    this.state = {
      train_limit: 350
    }
    this.onChangeTrainLimit = this.onChangeTrainLimit.bind(this)
  }

  onChangeTrainLimit(limit) {
    this.setState({train_limit: limit})
  }

  render() {
    const { classifier } = this.props
    if (!classifier) { return null }
    return (
      <div>
        <h2>{classifier.title}</h2>
        <h5><ClassifierDetails classifier={classifier} /></h5>

        <h3>Training</h3>
        <Row>
          <Col xs={12} sm={7} md={7} className='hspace'>
            <ButtonToolbar>
              <ButtonGroup>
                <Button onTouchTap={() => this.props.onTrainClicked()}>Train</Button>
              </ButtonGroup>
              <ButtonGroup>
                <Button onTouchTap={() => this.props.onTrainClicked(this.state.train_limit)}>Train until</Button>
                <DropdownButton id='training_limit_dropdown' title={this.state.train_limit} onSelect={this.onChangeTrainLimit}>
                  {[10, 100, 200, 350, 500, 750, 1000, 2000, 3000].map(limit =>
                    <MenuItem key={limit} eventKey={limit}>{limit}</MenuItem>
                  )}
                </DropdownButton>
              </ButtonGroup>
              <ButtonGroup>
                <Button onTouchTap={() => this.props.onLearnClicked()}>Learn</Button>
              </ButtonGroup>
            </ButtonToolbar>

            <div className="hspace">
              Train a Classifier to improvie it's Performance. "Train" will iterate random samples until no samples are left.
              "Learn" will query the most interesting messages and ask you to review them.
            </div>
          </Col>
          <Col xs={12} sm={5} md={5}>
            <Review classifier={classifier} />
          </Col>
        </Row>

        <ClassifierPerformance classifier={classifier} />
      </div>
    )
  }
}

TrainingContainer.propTypes = {
  classifier: PropTypes.any,
  onTrainClicked: PropTypes.func.isRequired,
  onLearnClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    classifier: classifierById(state, ownProps.params.classifier_id)
  }
}

const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    onTrainClicked: (train_limit) => {
      dispatch(trainClassifier(ownProps.params.classifier_id, train_limit))
    },
    onLearnClicked: () => {
      dispatch(learnClassifier(ownProps.params.classifier_id))
    }
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(TrainingContainer)
