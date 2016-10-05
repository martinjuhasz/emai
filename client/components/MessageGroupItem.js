import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import {ListGroupItem } from 'react-bootstrap/lib'
import Emoticon from './Emoticon'

class MessageGroupItem extends Component {

  getListStyle(message, selected_message) {
    if (selected_message && selected_message === (message._id || message.id)) {
      return 'info'
    }
    if ('label' in message) {
      return this.getStyle(message.label)
    }
    return null
  }

  getStyle(label) {
    switch (label) {
      case 1:
        return 'warning'
      case 2:
        return 'danger'
      case 3:
        return 'success'
      default:
        return null
    }
  }

  renderPredictedLabel(message) {
    if(!message.predicted_label) { return null }
    const label = this.getStyle(message.predicted_label)
    const style = `prediction-label ${label}`
    return (
      <div className={style}></div>
    )
  }

  render() {
    const { message, selected_message } = this.props
    const style = this.getListStyle(message, selected_message)
    return (
      <ListGroupItem bsStyle={style} onTouchTap={this.props.onTouchTap}>
        {this.renderPredictedLabel(message)}
        {message.content}
        {message.emoticons.map((emoticon) => <Emoticon key={emoticon.identifier} emoticon={emoticon}/>)}
      </ListGroupItem>
    )
  }
}

MessageGroupItem.propTypes = {
  message: PropTypes.any.isRequired,
  selected_message: PropTypes.string,
  onTouchTap: PropTypes.func
}


export default connect(

)(MessageGroupItem)
