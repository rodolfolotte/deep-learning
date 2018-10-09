import sys, os
import osgeo.osr as osr, ogr
import json
import gdal
import shapefile

# USAGE: python raster2shp.py PATH_TO_IMAGE SAVE_DESTINATION

def create_polygon(json):          
    # ring = ogr.Geometry(ogr.wkbLinearRing)

    # # for coord in json.['features'][0]['geometry']['coordinates']:
    # for coord in json:
    #     ring.AddPoint(coord[0], coord[1])
    
    # poly = ogr.Geometry(ogr.wkbPolygon)
    # poly.AddGeometry(ring)
    # geom = ogr.CreateGeometryFromWkt(poly.ExportToWkt())

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(-55.11453010733062, -4.269241210699577)
    ring.AddPoint(-54.88656945846435, -4.314357119070749)
    ring.AddPoint(-54.90070153049529, -4.388320447467832)
    ring.AddPoint(-55.12902153418034, -4.343037768916627)
    ring.AddPoint(-55.11678178557853, -4.279836579969911)
    ring.AddPoint(-55.116600965728175, -4.279872438091217)
    ring.AddPoint(-55.11453010733062, -4.269241210699577)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)    
    geom = ogr.CreateGeometryFromWkt(poly.ExportToWkt())   

    return geom

def create_shapefile(geom, image, json):
    image_projection = image.GetProjectionRef()    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(shapefile)
    srs = osr.SpatialReference(wkt=image_projection)     
    srs.ImportFromEPSG(4326)
    
    layer = ds.CreateLayer('background', srs, ogr.wkbPolygon)

    _id = ogr.FieldDefn('id', ogr.OFTInteger)        
    _area = ogr.FieldDefn('area', ogr.OFTReal)
    _image_id = ogr.FieldDefn('image-id', ogr.OFTString)
    _image_item_type = ogr.FieldDefn('item-type', ogr.OFTString)
    _image_date = ogr.FieldDefn('image-date', ogr.OFTString)
    _image_cloud_cover = ogr.FieldDefn('cloud-cov', ogr.OFTReal)

    layer.CreateField(_id)
    layer.CreateField(_area)
    layer.CreateField(_image_id)
    layer.CreateField(_image_item_type)
    layer.CreateField(_image_date)
    layer.CreateField(_image_cloud_cover)

    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)

    feature.SetGeometry(geom)   
    feature.SetField('id', 1)
    feature.SetField('area', 0.0521)
    feature.SetField('image-id', json_metadata['features'][0]['id'])
    feature.SetField('item-type', json_metadata['features'][0]['properties']['item_type'])
    feature.SetField('image-date', json_metadata['features'][0]['properties']['acquired'])
    feature.SetField('cloud-cov', json_metadata['features'][0]['properties']['cloud_cover'])

    layer.CreateFeature(feature)


if __name__=='__main__':
    image_path = sys.argv[1]
    output = sys.argv[2]

    path, filename = os.path.split(os.path.abspath(image_path))
    name, extension = os.path.splitext(filename)
    metadata = path + "/" + name + ".json"
    shapefile = path + "/" + name + ".shp"

    if(not os.path.isfile(metadata)):
        print "Metadata related to the image " + name + " does not exist. Check it and try again!"
        sys.exit(1)
    json_metadata = json.load(open(metadata))       
    
    image = gdal.Open(image_path, gdal.GA_ReadOnly)
    if image is None:
        print 'Could not open image file!'
        sys.exit(1)

    geom = create_polygon(json_metadata)
    create_shapefile(geom, image, json_metadata)   