import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import ClassifierResult from './ClassifierResult';
import SettingsResult from './SettingsResult';
import Probes from './Probes';
import {ListGroupItem } from 'react-bootstrap/lib'
import Emoticon from './Emoticon'

class MessageGroupItem extends Component {

  getListStyle(message, selected_message) {
  	if(selected_message && selected_message === message._id) {
  		return 'info'
  	}
  	if('label' in message) {
  		switch(message.label) {
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
  	return null
  }

  render() {
    const { message, selected_message } = this.props
    const style = this.getListStyle(message, selected_message)
    return (
    	<ListGroupItem bsStyle={style} onTouchTap={this.props.onTouchTap}>
    		{message.content}
    		{message.emoticons.map((emoticon) => <Emoticon key={emoticon.identifier} emoticon={emoticon} />)}
    	</ListGroupItem>
    )
  }
}

MessageGroupItem.propTypes = {
  message: PropTypes.any.isRequired,
  selected_message: PropTypes.string
}


export default connect(

)(MessageGroupItem)