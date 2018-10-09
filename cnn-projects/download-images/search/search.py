import os
import requests

from requests.auth import HTTPBasicAuth

from parameters.constants import API_KEY
from parameters.filters import redding_reservoir

stats_endpoint_request = {
  "interval": "day",
  "item_types": ["REOrthoTile"],
  "filter": redding_reservoir
}

result = \
  requests.post(
    'https://api.planet.com/data/v1/stats',
    auth=HTTPBasicAuth(os.environ[API_KEY], ''),
    json=stats_endpoint_request)

print result.text


