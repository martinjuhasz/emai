import fetch from 'isomorphic-fetch'
import { Schema, arrayOf, normalize } from 'normalizr'

const message = new Schema('message', {idAttribute: '_id'});
const sample = new Schema('sample');
sample.define({
  messages: arrayOf(message)
})

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

  startRecording(username, callback) {
    return fetch(`${api_url}/recordings`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        channel: username
      })
    }).then(() => callback())
  },

  stopRecording(recording_id, callback) {
    const url = `${api_url}/recordings/${recording_id}/stop`
    return fetch(url, {
      method: 'PUT',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    })
      .then(() => callback())
  },

  deleteRecording(recording_id, callback) {
    return fetch(`${api_url}/recordings/${recording_id}`, {
      method: 'DELETE'
    })
      .then(() => callback())
  },

  getClassifiers(callback) {
    const url = `${api_url}/classifiers`
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(json))
  },

  createClassifier(title, callback) {
    return fetch(`${api_url}/classifiers`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        title: title
      })
    }).then(() => callback())
  },

  deleteClassifier(classifier_id, callback) {
    return fetch(`${api_url}/classifiers/${classifier_id}`, {
      method: 'DELETE'
    })
      .then(() => callback())
  },

  trainClassifier(classifier_id, limit, callback) {
    const url = `${api_url}/classifiers/${classifier_id}/train?train_count=${limit}`
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
  },

  getMessagesAtTime(recording, time, last_message, classifier, callback) {
    return fetch(`${api_url}/recordings/${recording}/messages/${time}?last_message=${last_message}&classifier=${classifier}`)
      .then(response => {
        if (response.status === 204) return []
        return response.json()
      })
      .then(json => callback(json))
  }

}


