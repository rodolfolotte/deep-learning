import os, sys
import datetime 
import json

def getFilterSetupById(id):
  # filter images acquired in a certain date range
  search_id_filter = {  
   "type":"StringInFilter",
   "field_name":"id",
   "config":[id]
  }

  # create a filter that combines our geo and date filters could also use an "OrFilter"
  image_filter_setup = {
    "type": "AndFilter",
    "config": [search_id_filter]
  }
  
  return image_filter_setup

def getFilterSetup(perimeter, date_1, date_2, cloud_coverage):
  # filter for items the overlap with our chosen geometry
  with open(perimeter) as content_file:
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": json.load(content_file)
    }

  # filter images acquired in a certain date range
  date_range_filter = {
    "type": "DateRangeFilter",
    "field_name": "acquired",
    "config": {
      "gte": date_1,
      "lte": date_2
    }
  }

  # filter any images which are more than 50% clouds
  cloud_cover_filter = {
    "type": "RangeFilter",
    "field_name": "cloud_cover",
    "config": {
      "gte": 0.0,
      "lte": cloud_coverage
    }
  }

  # create a filter that combines our geo and date filters could also use an "OrFilter"
  image_filter_setup = {
    "type": "AndFilter",
    "config": [geometry_filter, date_range_filter, cloud_cover_filter]
  }
  
  return image_filter_setup




