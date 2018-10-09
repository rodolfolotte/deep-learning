import os, sys
import requests
import time
import json
import urllib

import parameter.constants as constants
import parameter.filters as filters

from requests.auth import HTTPBasicAuth
from os.path import isfile, join
from os import listdir

asset_type = 'analytic_sr'
onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

session = requests.Session()
session.auth = (constants.API_KEY, '')

# first, activate the images
# for id in onlyfiles:
#     lines = [line.rstrip('\n\r') for line in open('id/' + id)]
#
#     for item in lines:
#         if(item!=''):
#             print('Requesting ' + item + '...')
#             activate_item = session.get(constants.DOWNLOAD + '/' + constants.PLANET_4BAND + '/items/' + item + '/assets/')
#
#             if(asset_type in activate_item.json()):
#                 activate_item_url = activate_item.json()[asset_type]["_links"]["activate"]
#                 response_item = session.post(activate_item_url)
#             else:
#                 print('The key ' + asset_type + ' is not available for the id ' + item)

# second, download if the image is active
for id in onlyfiles:
    lines = [line.rstrip('\n\r') for line in open('id/' + id)]

    for item in lines:
        if(item!=''):
            endpoint_search = {    
                "item_types": [constants.PLANET_4BAND],
                "filter": filters.getFilterSetupById(item)
            }
        
            activate_item = session.get(constants.DOWNLOAD + '/' + constants.PLANET_4BAND + '/items/' + item + '/assets/')

            if(asset_type in activate_item.json()):

                while(activate_item.json()[asset_type]['status'] == 'inactive'):
                    print("Still inactive... lets wait a minute more!")
                    time.sleep(60)

                print("Status: " + activate_item.json()[asset_type]['status'])
                print('Downloading: ' + item + '...')
                properties_item = session.post(constants.SEARCH, json=endpoint_search)    

                if(activate_item.json()[asset_type]['location'] == '' or activate_item.json()[asset_type]['location'] == None):
                    print('Tag \'location\' not found. Download skipped for ' + item + '. ')
                else:
                    urllib.urlretrieve(activate_item.json()[asset_type]['location'], 'images/' + item + '.tif')
                    
                with open('images/' + item + '.json', 'w') as outf:                    
                    json.dump(properties_item.json(), outf)    
            else:
                print('Download skipped for ' + item + '. ' + asset_type + ' not available.')            


