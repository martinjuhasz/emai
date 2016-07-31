import fetch from 'isomorphic-fetch'
import { Schema, arrayOf, normalize } from 'normalizr'

const message = new Schema('message', { idAttribute: '_id' });
const sample = new Schema('sample');
sample.define({
  messages: arrayOf(message)
})

const api_url = 'http://10.0.1.88:8082'
/* const api_url = 'http://0.0.0.0:8082' */

export default {
  
  getSamples(recording_id, interval, callback) {
    const url = `${api_url}/recordings/${recording_id}/samples/${interval}`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(normalize(json, arrayOf(sample))))
  },

  classifySample(sample_id, label, hiddenMessages) {
  	return fetch(`${api_url}/samples/${sample_id}`, {
  	  method: 'PUT',
  	  headers: {
  	    'Accept': 'application/json',
  	    'Content-Type': 'application/json'
  	  },
  	  body: JSON.stringify({
  	    label: label,
        hidden: hiddenMessages
  	  })
    })
  },

  classifyMessages(messages) {
    console.log('pew')
  },

  getRecordings(callback) {
  	return fetch(`${api_url}/recordings`)
      .then(response => response.json())
      .then(json => callback(json))
  },

  getClassifiers(recording_id, callback) {
    const url = `${api_url}/training/${recording_id}`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(json))
  },

  
}


