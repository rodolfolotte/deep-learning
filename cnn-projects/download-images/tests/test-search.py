import os, sys
import requests
import json
import urllib

import parameter.constants as constants
import parameter.filters as filters

from requests.auth import HTTPBasicAuth
from os.path import basename

# Planet examples 
# https://www.planet.com/docs/api-quickstart-examples/step-2-download/

# Github Planet API
# https://planetlabs.github.io

# command line examples
# https://planetlabs.github.io/planet-client-python/cli/index.html

# original = sys.argv[1]
# reference = sys.argv[2]

# original_base = basename(original)
# reference_base = basename(reference)

session = requests.Session()
session.auth = (constants.API_KEY, '')

date1 = '2017-07-07T00:00:00.000Z'
date2 = '2017-07-08T00:00:00.000Z'
cloud_percentage = 0.5
asset_type = 'analytic_sr'

for aoi in range(1, int(constants.NUM_OF_AOIS)+1):
  endpoint_search = {
    "interval": "day",
    "item_types": [constants.PLANET_4BAND],
    "filter": filters.getFilterSetup('aoi/aoi' + str(aoi) + '.geojson', date1, date2, cloud_percentage)
  }

  endpoint_download = {
    "item_types": [constants.PLANET_4BAND],
    "filter": filters.getFilterSetup('aoi/aoi' + str(aoi) + '.geojson', date1, date2, cloud_percentage)
  }

  response = session.post(constants.SEARCH, json=endpoint_search)
  json_response = response.json()

  # first, activate the images
  for item in json_response['features']:
    # TODO: verificar primeiro se ha asset (se tem autorizacao para download)
    activate_item = session.get(constants.DOWNLOAD + "/" + constants.PLANET_4BAND + "/items/" + item['id'] + "/assets/")
    activate_item_url = activate_item.json()[asset_type]["_links"]["activate"]
    response_item = session.post(activate_item_url)
        
  # second, download if the image is active
  for item in json_response['features']:
    if not os.path.exists('images/' + date1):
      os.makedirs('images/' + date1)
    
    if(activate_item.json()[asset_type]['status'] == 'active'):      
      # session.get(activate_item.json()[asset_type]['location'])
      urllib.urlretrieve(activate_item.json()[asset_type]['location'], 'images/' + date1 + '/' + item['id'] + '.tif')
          
      with open('images/' + date1 + '/' + item['id'] + '.json', 'w') as outf:
        json.dump(item, outf)