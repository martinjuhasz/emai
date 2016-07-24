const HtmlWebpackPlugin = require('html-webpack-plugin');
const TransferWebpackPlugin = require('transfer-webpack-plugin');
const path = require('path');

module.exports = {
  devtool: '#inline-source-map',
  entry: './client/index.js',
  output: {
    path: 'static',
    filename: './assets/app.js'
  },
  module: {
    loaders: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel',
        query: {
          presets: ['es2015', 'react']
        }
      }
    ]
  },
  plugins: [
    new TransferWebpackPlugin([
      {from: '../client/static'},
    ], path.resolve(__dirname, 'static')),
  ]
};
