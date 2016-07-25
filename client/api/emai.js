import fetch from 'isomorphic-fetch'
import { Schema, arrayOf, normalize } from 'normalizr'

const message = new Schema('message', { idAttribute: '_id' });
const sample = new Schema('sample');
sample.define({
  messages: arrayOf(message)
})

export default {
  
  getSamples(recording_id, data_set, callback) {
    const url = `http://localhost:8080/recordings/${recording_id}/data-sets/${data_set}/sample`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(normalize(json, arrayOf(sample))))
  },

  classifySample(sample_id, label, hiddenMessages) {
  	return fetch(`http://localhost:8080/samples/${sample_id}`, {
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

  getRecordings(callback) {
  	return fetch('http://localhost:8080/recordings')
      .then(response => response.json())
      .then(json => callback(json))
  }
}


