import React, {Component, PropTypes} from 'react'
import {connect} from 'react-redux'
import {Line as LineChart} from 'react-chartjs'
import times from 'lodash/times'
import {Panel} from 'react-bootstrap/lib'
import {ButtonToolbar, ButtonGroup, Button, Glyphicon } from 'react-bootstrap/lib'

class ClassifierResultChart extends Component {

  constructor() {
    super()
    this.state = {
      selected_performance: 'precision'
    }
    this.handlePerformanceStateClick = this.handlePerformanceStateClick.bind(this)
    this.getDataForState = this.getDataForState.bind(this)
    this.isPerformanceSelected = this.isPerformanceSelected.bind(this)
  }

  handlePerformanceStateClick(state) {
    this.setState({selected_performance: state})
  }

  isPerformanceSelected(performance) {
    if(!this.state) {
      return false
    }
    return this.state.selected_performance === performance;
  }

  getDataForState(state) {
    const {classifier} = this.props
    if (!classifier || !classifier.performance) {
      return null
    }

    const labels = times(classifier.performance.negative.time.length, String)
    return {
      labels: labels,
      datasets: this.getDataSetsForState(classifier, state)
    }
  }

  getDataSetsForState(classifier, state) {
    switch(state) {
      case 'precision': {
        return [
          Object.assign(
            {
              data: classifier.performance.negative.precision,
              label: 'Negative Precision',
            },
            this.getChartStyling(0)
          ),
          Object.assign(
            {
              data: classifier.performance.positive.precision,
              label: 'Positive Precision',
            },
            this.getChartStyling(1)
          ),
          Object.assign(
            {
              data: classifier.performance.neutral.precision,
              label: 'Neutral Precision',
            },
            this.getChartStyling(2)
          )
        ]
      }
      case 'f1': {
        return [
          Object.assign(
            {
              data: classifier.performance.negative.fscore,
              label: 'Negative F1 Score',
            },
            this.getChartStyling(0)
          ),
          Object.assign(
            {
              data: classifier.performance.positive.fscore,
              label: 'Positive F1 Score',
            },
            this.getChartStyling(1)
          ),
          Object.assign(
            {
              data: classifier.performance.neutral.fscore,
              label: 'Neutral F1 Score',
            },
            this.getChartStyling(2)
          )
        ]
      }
      case 'recall': {
        return [
          Object.assign(
            {
              data: classifier.performance.negative.recall,
              label: 'Negative Recall',
            },
            this.getChartStyling(0)
          ),
          Object.assign(
            {
              data: classifier.performance.positive.recall,
              label: 'Positive Recall',
            },
            this.getChartStyling(1)
          ),
          Object.assign(
            {
              data: classifier.performance.neutral.recall,
              label: 'Neutral Recall',
            },
            this.getChartStyling(2)
          )
        ]
      }
      case 'support': {
        return [
          Object.assign(
            {
              data: classifier.performance.negative.support,
              label: 'Negative Support',
            },
            this.getChartStyling(0)
          ),
          Object.assign(
            {
              data: classifier.performance.positive.support,
              label: 'Positive Support',
            },
            this.getChartStyling(1)
          ),
          Object.assign(
            {
              data: classifier.performance.neutral.support,
              label: 'Neutral Support',
            },
            this.getChartStyling(2)
          )
        ]
      }
      default:
        return null
    }
  }

  getChartStyling(type) {
    switch(type) {
      case 0:
        return {
          fill: false,
          lineTension: 0.1,
          borderCapStyle: 'butt',
          borderColor: "rgba(217,83,79,0.7)",
          backgroundColor: "rgba(217,83,79,0.2)",
        }
      case 1:
        return {
          fill: false,
          lineTension: 0.1,
          borderCapStyle: 'butt',
          borderColor: "rgba(92,184,92,0.7)",
          backgroundColor: "rgba(92,184,92,0.2)",
        }
      case 2:
        return {
          fill: false,
          lineTension: 0.1,
          borderCapStyle: 'butt',
          borderColor: "rgba(240,173,78,0.7)",
          backgroundColor: "rgba(240,173,78,0.2)",
        }
    }
  }

  render() {
    const {classifier} = this.props
    if(!classifier || !classifier.performance) {
      return null
    }

    const options = {
      legend: {
        position: 'bottom'
      }
    }

    return (
      <div>
        <ButtonToolbar className='hspace'>
          <ButtonGroup>
            <Button onTouchTap={() => this.handlePerformanceStateClick('precision')}>Precision</Button>
            <Button onTouchTap={() => this.handlePerformanceStateClick('recall')}>Recall</Button>
            <Button onTouchTap={() => this.handlePerformanceStateClick('f1')}>F-beta score</Button>
          </ButtonGroup>
        </ButtonToolbar>
        <Panel>
          <LineChart data={this.getDataForState(this.state.selected_performance)} options={options}/>
        </Panel>

        {this.isPerformanceSelected('precision') && <div>
          The <strong>precision</strong> is the ratio of true positives / (true positives + false positives). The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        </div>}
        {this.isPerformanceSelected('f1') && <div>
          The <strong>F-beta score</strong> can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
          The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
        </div>}
        {this.isPerformanceSelected('recall') && <div>
          The <strong>recall</strong> is the ratio of true positives / (true positives + false negatives). The recall is intuitively the ability of the classifier to find all the positive samples.
        </div>}
        {this.isPerformanceSelected('support') && <div>
          The <strong>support</strong> is the number of occurrences of each class.
        </div>}

      </div>
    )
  }
}

ClassifierResultChart.propTypes = {
  classifier: PropTypes.any
}


export default connect(

)(ClassifierResultChart)
