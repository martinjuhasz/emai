import { combineReducers } from 'redux'
import * as types from '../constants/ActionTypes'
import merge from 'lodash/merge'
import unset from 'lodash/unset'
import pullAll from 'lodash/pullAll'
import indexOf from 'lodash/indexOf'
import find from 'lodash/find'

export function all(state = [], action) {
  switch (action.type) {
    case types.RECEIVE_CLASSIFIERS:
      return action.classifiers
    case types.RECEIVE_CLASSIFIER: {
      const new_state = merge([], state)
      const index = indexOf(new_state, find(new_state, {id: action.classifier.id}))
      new_state[index] = action.classifier
      return new_state
    }
    default:
      return state
  }
}

export function reviews(state = {}, action) {
  switch (action.type) {
    case types.RECEIVE_REVIEW: {
      return merge({}, state, {
        [action.classifier]: action.reviews.result
      })
    }
    case types.SAVE_REVIEW: {
      const message_ids = action.messages.map(message => message.id)
      const new_state = merge({}, state)
      pullAll(new_state[action.classifier], message_ids)
      return new_state
    }
    default:
      return state
  }
}

export function messages(state = {}, action) {
  switch (action.type) {
    case types.RECEIVE_REVIEW:
      return action.reviews.entities.review
    case types.CLASSIFY_REVIEW:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: action.label})
      })
    case types.DECLASSIFY_REVIEW:
      return merge({}, state, {
        [action.message]: merge({}, state[action.message], {label: null})
      })
    case types.SAVE_REVIEW: {
      const new_state = merge({}, state)
      for (const message of action.messages) {
        unset(new_state, message.id)
      }
      return new_state
    }
    default:
      return state
  }
}

export function byId(state, classifier_id) {
  return state.classifiers.all.find(classifier => classifier.id === classifier_id)
}

export function getReviews(state, classifier_id, limit=10) {
  const message_ids = state.classifiers.reviews[classifier_id]
  if (!message_ids || message_ids.length <= 0) {
    return null
  }
  return state.classifiers.reviews[classifier_id].map(mid => state.classifiers.messages[mid]).slice(0, limit)
}

export default combineReducers({
  all,
  reviews,
  messages
})
