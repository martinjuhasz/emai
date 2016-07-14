from flask import Flask


class APIServer(object):
    def __init__(self):
        self.flask = Flask(__name__)

    def start(self):
        self.flask.run()