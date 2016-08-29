import fetch from 'isomorphic-fetch'
import { Schema, arrayOf, normalize } from 'normalizr'

const message = new Schema('message', {idAttribute: '_id'});
const sample = new Schema('sample');
sample.define({
  messages: arrayOf(message)
})
const review = new Schema('review');

/* const api_url = 'http://10.0.1.88:8082' */
const api_url = 'http://0.0.0.0:8082'

export default {

  getSamples(recording_id, interval, callback) {
    const url = `${api_url}/recordings/${recording_id}/samples/${interval}`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(normalize(json, arrayOf(sample))))
  },

  classifyMessages(messages, callback) {
    const json_messages = messages.filter(message => message.label > 0).map(message => {
      return {'id': (message._id || message.id), 'label': message.label}
    })
    if (!json_messages || json_messages.length <= 0) {
      return
    }

    return fetch(`${api_url}/messages`, {
      method: 'PUT',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        messages: json_messages
      })
    })
      .then(() => callback())
  },

  getRecordings(callback) {
    return fetch(`${api_url}/recordings`)
      .then(response => response.json())
      .then(json => callback(json))
  },

  getClassifiers(callback) {
    const url = `${api_url}/classifiers`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(json))
  },

  getReview(classifier_id, callback) {
    const url = `${api_url}/classifiers/${classifier_id}/review`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(normalize(json, arrayOf(review))))
  },

  trainClassifier(classifier_id, callback) {
    const url = `${api_url}/classifiers/${classifier_id}/train`
    return fetch(url,  {method: 'POST'})
      .then(response => response.json())
      .then(json => callback(json))
  },

  updateClassifier(classifier_id, settings, type, callback) {
    const url = `${api_url}/classifiers/${classifier_id}`
    return fetch(url, {
      method: 'PUT',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        type: type,
        settings: settings
      })
    })
      .then(response => response.json())
      .then(json => callback(json))
  },

  learnClassifier(classifier_id, callback) {
    const url = `${api_url}/classifiers/${classifier_id}/learn`
    return fetch(url,  {method: 'POST'})
      .then(response => response.json())
      .then(json => callback(json))
  }


}


