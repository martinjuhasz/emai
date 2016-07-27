import React, { Component, PropTypes } from 'react'

export default class Message extends Component {
  render() {
    const { message } = this.props
    return (
    	<div onTouchTap={this.props.onTouchTap}>
			<strong>User:</strong> {message.content} 
			{message.emoticons.map(emoticon => {
				const url = `http://static-cdn.jtvnw.net/emoticons/v1/${emoticon.identifier}/1.0`
				return <img src={url} key={emoticon.identifier} />
			})}
		</div>
	)
  }
}

Message.propTypes = {
  message: PropTypes.shape({
    user_id: PropTypes.string.isRequired,
    content: PropTypes.string.isRequired,
    username: PropTypes.string,
    created: PropTypes.string.isRequired,
    emoticons: PropTypes.array
  })
}
