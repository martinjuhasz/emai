import React, { Component, PropTypes } from 'react'
import { connect } from 'react-redux'
import ClassifierResult from './ClassifierResult';
import SettingsResult from './SettingsResult';
import Probes from './Probes';


class Emoticon extends Component {
  render() {
    const { emoticon } = this.props
    const url = `http://static-cdn.jtvnw.net/emoticons/v1/${emoticon.identifier}/1.0`
    return (
      <img className='emoticon' src={url} />
    )
  }
}

Emoticon.propTypes = {
  emoticon: PropTypes.any.isRequired
}


export default connect(

)(Emoticon)
