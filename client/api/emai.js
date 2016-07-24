import fetch from 'isomorphic-fetch'

export default {
  
  getSamples(recording_id, data_set, callback) {
    const url = `http://localhost:8080/recordings/${recording_id}/data-sets/${data_set}/sample`
    console.log(url)
    return fetch(url)
      .then(response => response.json())
      .then(json => callback(json))
  },

  classifySample(sample_id, label) {
  	return fetch(`http://localhost:8080/samples/${sample_id}`, {
	  method: 'PUT',
	  headers: {
	    'Accept': 'application/json',
	    'Content-Type': 'application/json'
	  },
	  body: JSON.stringify({
	    label: label
	  })
	})
  },

  getRecordings(callback) {
  	return fetch('http://localhost:8080/recordings')
      .then(response => response.json())
      .then(json => callback(json))
  }
}


