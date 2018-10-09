import os, sys
import gdal
import osgeo.osr as osr, ogr
import numpy as np
import argparse
import cv2

from os.path import basename

# USAGE: python segment-polygon.py -input_folder PATH_TO_IMAGE -output_folder PATH_TO_OUTPUT
# EXAMPLE: python segment-polygon.py -input_folder /data/personal/sccon/dataset/test/ -output_folder /data/personal/sccon/dataset/test2/

classes = {
    "def": [255, 255, 0],
    "water": [102, 204, 255],
    "cloud": [255, 255, 255],
    "shadow": [128, 128, 128]
}

def createGeometries(corners, hierarchy, image):
    gt = image.GetGeoTransform()
    geom = []
    for i in range(len(corners)):
        flag = True
        ring = ogr.Geometry(ogr.wkbLinearRing)

        if(hierarchy[0,i,3] == -1):
            for coord in corners[i]:
                Xgeo = gt[0] + coord[0][0] * gt[1] + coord[0][1] * gt[2]
                Ygeo = gt[3] + coord[0][0] * gt[4] + coord[0][1] * gt[5]
                ring.AddPoint(Xgeo, Ygeo)

                if (flag == True):
                    flag = False
                    initialX = Xgeo
                    initialY = Ygeo

            ring.AddPoint(initialX, initialY)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            geom.append(ogr.CreateGeometryFromWkt(poly.ExportToWkt()))

    return geom


def getClassesGT(complete_path_png):
    imageSegmented = cv2.imread(complete_path_png)
    imageSegmented = cv2.cvtColor(imageSegmented, cv2.COLOR_BGR2RGB)

    height, width, bands = imageSegmented.shape

    # TODO: recurso .any() e .all() nao funcionou adequadamente. Acusa ter uma cor de pixel, que na realidade nao existe na imagem isso se deve ao fato de que ele compara as bandas individualmente
    # for key, value in classes.items():
    #     if ((imageSegmented == value).any()):
    #         print("yes " + key)
    #     else:
    #         print("no " + key)

    gt_classes = []
    for key, value in classes.items():
        for i in range(width):
            for j in range(height):
                if ((imageSegmented[i, j][0] == value[0]) and (imageSegmented[i, j][1] == value[1]) and (imageSegmented[i, j][2] == value[2]) and (key not in gt_classes)):
                    gt_classes.append(key)

    return gt_classes


def getImageByClass(complete_path_png, key):
    imageSegmented = cv2.imread(complete_path_png)
    imageSegmented = cv2.cvtColor(imageSegmented, cv2.COLOR_BGR2RGB)

    value = classes[key]
    height, width, bands = imageSegmented.shape

    for i in range(width):
        for j in range(height):
            if not ((imageSegmented[i, j][0] == value[0]) and (imageSegmented[i, j][1] == value[1]) and (imageSegmented[i, j][2] == value[2])):
                imageSegmented[i, j][0] = 0
                imageSegmented[i, j][1] = 0
                imageSegmented[i, j][2] = 0

    return imageSegmented



def createShapefile(complete_path, complete_path_png, complete_path_shp, json_metadata):
    filename = basename(complete_path)
    name = os.path.splitext(filename)[0]

    image = gdal.Open(complete_path)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(complete_path_shp)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(image.GetProjection())

    _id = ogr.FieldDefn('id', ogr.OFTInteger)
    _area = ogr.FieldDefn('area', ogr.OFTReal)
    _class = ogr.FieldDefn('class', ogr.OFTString)
    _image_id = ogr.FieldDefn('image-id', ogr.OFTString)
    _image_item_type = ogr.FieldDefn('item-type', ogr.OFTString)
    _image_date = ogr.FieldDefn('image-date', ogr.OFTString)
    _image_cloud_cover = ogr.FieldDefn('cloud-cov', ogr.OFTReal)

    gt_classes = getClassesGT(complete_path_png)

    classesAndGeometries = {}
    for k in range(len(gt_classes)):
        imageSegmented = getImageByClass(complete_path_png, gt_classes[k])
        imageSegmentedInGray = cv2.cvtColor(imageSegmented, cv2.COLOR_RGB2GRAY)

        thresh = cv2.threshold(imageSegmentedInGray, 127, 255, 0)[1]
        im2, corners, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        geometries = createGeometries(corners, hierarchy, image)

        classesAndGeometries[gt_classes[k]] = geometries

    layer = ds.CreateLayer(name, srs, ogr.wkbPolygon)

    layer.CreateField(_id)
    layer.CreateField(_area)
    layer.CreateField(_class)
    layer.CreateField(_image_id)
    layer.CreateField(_image_item_type)
    layer.CreateField(_image_date)
    layer.CreateField(_image_cloud_cover)

    for key, value in classesAndGeometries.items():
        for g in range(len(value)):
            cont = 1
            featureDefn = layer.GetLayerDefn()
            feature = ogr.Feature(featureDefn)

            feature.SetGeometry(value[g])
            feature.SetField('id', cont)
            feature.SetField('area', value[g].GetArea())
            feature.SetField('class', key)

            feature.SetField('image-id', '25123352_15242')
            feature.SetField('item-type', 'PSScene4Band')
            feature.SetField('image-date', '2017-07-02 11:30:00')
            feature.SetField('cloud-cov', '0.25')
            # feature.SetField('image-id', json_metadata['features'][0]['id'])
            # feature.SetField('item-type', json_metadata['features'][0]['properties']['item_type'])
            # feature.SetField('image-date', json_metadata['features'][0]['properties']['acquired'])
            # feature.SetField('cloud-cov', json_metadata['features'][0]['properties']['cloud_cover'])

            layer.CreateFeature(feature)

            cont += 1


def segmentImage(imageFolder, outputFolder):
    valid_images = [".tif", ".geotiff", ".tiff"]

    for f in os.listdir(imageFolder):
        ext = os.path.splitext(f)[1]

        if ext.lower() not in valid_images:
            continue

        complete_path = os.path.join(imageFolder, f)
        filename = basename(complete_path)
        name, file_extension = os.path.splitext(filename)

        complete_path_shp = outputFolder + name + ".shp"
        complete_path_png = imageFolder + name + ".png"
        complete_path_json = imageFolder + name + ".json"

        # if((not os.path.isfile(complete_path)) or (not os.path.isfile(complete_path_json))):
        if ((not os.path.isfile(complete_path))):
            print(">> Image or JSON is not present for file: " + name + ". Check it and try again!")
            continue

        if ((not os.path.isfile(complete_path_png))):
            print(">> Image PNG is not present for file: " + complete_path_png + ". Check it and try again!")
            continue

        # if ((not os.path.isfile(complete_path_shp))):
        #     print(">> Shapefile is not present for file: " + complete_path_shp + ". Check it and try again!")
        #     continue

        # TODO: incluir segmentacao da imagem pela CNN. O retorno sera o png com as predicoes

        createShapefile(complete_path, complete_path_png, complete_path_shp, complete_path_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blablablabla')
    parser.add_argument('-input_folder', action="store", dest='imageFolder', help='Image folder')
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Output folder')

    result = parser.parse_args()

    segmentImage(result.imageFolder, result.outputFolder)