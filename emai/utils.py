"""
Konfiguration von Logging und einem Config-Objekt welches die config.ini ausliest.
"""
import logging
import sys
from configparser import ConfigParser
import os

# Logging
output_stream = sys.stderr
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=output_stream)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

# Config
config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
