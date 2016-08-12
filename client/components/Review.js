import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import { classifyReview, saveReview, declassifyReview, classifyReviewMessage } from '../actions'
import Sample from '../components/Sample'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'

class Review extends Component {

  constructor() {
    super()
    this.state = {
      selected_message: null
    }
    this.handleMessageClick = this.handleMessageClick.bind(this)
    this.handleClassifyClick = this.handleClassifyClick.bind(this)
  }

  handleMessageClick(message_id) {
    this.setState({selected_message: message_id})
  }

  handleClassifyClick(label) {
    if(this.state.selected_message) {
       this.props.classifyReviewMessage(this.state.selected_message, label)
       this.setState({selected_message: null})
    } else {
      this.props.classifyReview(this.props.messages, label)
    }
  }

  render() {
    const { messages, classifier } = this.props

    return (
      <Col xs={12} sm={5} md={5}>
        <Col className='hspace'>
          <SampleToolbar
            onClassifyClicked={this.handleClassifyClick}
            onUndoClicked={() => this.props.declassifyReview(messages)}
            onSaveClicked={() => this.props.saveReview(classifier.id, messages)} />
        </Col>
        <Col>
          {messages &&
          <Sample
            messages={messages}
            onMessageClicked={(message_id) => { this.handleMessageClick(message_id) }}
            selected_message={this.state.selected_message} />
          }
        </Col>
      </Col>
    )
  }
}

Review.propTypes = {
  messages: PropTypes.any,
  classifier: PropTypes.any,
  params: PropTypes.any,
  classifyReview: PropTypes.func.isRequired,
  saveReview: PropTypes.func.isRequired,
  declassifyReview: PropTypes.func.isRequired,
  classifyReviewMessage: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {

  }
}

export default connect(
  mapStateToProps,
  {
    classifyReview,
    saveReview,
    declassifyReview,
    classifyReviewMessage
  }
)(Review)
