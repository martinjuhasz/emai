var HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  devtool: '#inline-source-map',
  entry: './client/index.js',
  output: {
    path: 'static',
    filename: 'js/index.js'
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
  plugins: [new HtmlWebpackPlugin({
    filename: 'index.html',
    template: 'client/index.html'
  })]
};
