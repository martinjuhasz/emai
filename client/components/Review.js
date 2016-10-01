import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import * as messages from '../actions/messages'
import Sample from '../components/Sample'
import {Col } from 'react-bootstrap/lib'
import SampleToolbar from '../components/SampleToolbar'
import { byId as messagesById } from '../reducers/messages'

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
       this.props.onClassifyClicked(this.state.selected_message, label)
       this.setState({selected_message: null})
    } else {
      this.props.onClassifyUnlabeledClicked(this.props.messages, label)
    }
  }

  render() {
    const { messages, classifier } = this.props

    return (
      <div>
        <Col className='hspace'>
          <SampleToolbar
            onClassifyClicked={this.handleClassifyClick}
            onUndoClicked={() => this.props.onDeclassifyClicked(messages)}
            onSaveClicked={() => this.props.onSaveClicked(classifier.id, messages)} />
        </Col>
        <Col>
          {messages &&
          <Sample
            messages={messages}
            onMessageClicked={(message_id) => { this.handleMessageClick(message_id) }}
            selected_message={this.state.selected_message} />
          }
        </Col>
      </div>
    )
  }
}

Review.propTypes = {
  messages: PropTypes.any,
  classifier: PropTypes.any,
  params: PropTypes.any,
  onClassifyUnlabeledClicked: PropTypes.func.isRequired,
  onSaveClicked: PropTypes.func.isRequired,
  onDeclassifyClicked: PropTypes.func.isRequired,
  onClassifyClicked: PropTypes.func.isRequired
}

function mapStateToProps(state, ownProps) {
  return {
    messages: messagesById(state, ownProps.classifier.unlabeled_train_set)
  }
}

export default connect(
  mapStateToProps,
  {
    onSaveClicked: messages.save,
    onClassifyClicked: messages.classify,
    onClassifyUnlabeledClicked: messages.classifyUnlabeled,
    onDeclassifyClicked: messages.declassify,
  }
)(Review)
