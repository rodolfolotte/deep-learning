from decouple import config

USER_SENTINEL_HUB = config('USER_SENTINEL_HUB', default='USER_SENTINEL_HUB')
PASS_SENTINEL_HUB = config('PASS_SENTINEL_HUB', default='PASS_SENTINEL_HUB')

VALID_RASTER_EXTENSION = (".jpg", ".png", ".tif", ".tiff")
VALID_VECTOR_EXTENSION = ".shp"

CLASS_NAME = 'CLASS_NAME'
CLASSES = {
    "FLORESTA": [133, 224, 133],
    "NAO_FLORESTA": [194, 194, 163]
}


