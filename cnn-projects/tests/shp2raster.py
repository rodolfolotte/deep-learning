# A script to rasterise a shapefile to the same projection & pixel resolution as a reference image.
# SOURCE: https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
import os, sys
import argparse
import gdal
import shapefile
import pngcanvas
import osgeo.osr as osr, ogr

from os.path import basename
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

classes = {    
    "def": [255,255,0]
}

# classes = {    
#     "def": [255,255,0],
#     "vegetation": [0,255,0],
#     "water": [0,0,255],
#     "cloud": [225,225,235],
#     "shadow": [80,80,124]
#   }

# SOURCE: https://github.com/GeospatialPython/geospatialpython/blob/master/shp2img.py
def createPNGfromSHP2(shapefile_name, outputFolder, iwidth, iheight):
    filename = basename(shapefile_name)    
    name, file_extension = os.path.splitext(filename)
    
    r = shapefile.Reader(shapefile_name)

    if not r:
        print('>> Error: could not open the shapefile')

    # rfields = [f[0] for f in r.fields[1:]]

    # for sr in r.shapeRecords():
    #     atr = dict(zip(rfields, sr.record))
    #     geom = sr.shape.__geo_interface__

    xdist = r.bbox[2] - r.bbox[0]
    ydist = r.bbox[3] - r.bbox[1]    
    xratio = iwidth/xdist
    yratio = iheight/ydist

    shapes = r.shapes()    
    fields = r.fields
    records = r.records()
    state_polygons = {}
    pixels = []

    img = Image.new("RGB", (iwidth, iheight), "black")
    draw = ImageDraw.Draw(img)

    for i, record in enumerate(records):
        if record[1] in classes:
            for x,y in shapes[i].points:        
                px = int(iwidth - ((r.bbox[2] - x) * xratio))
                py = int((r.bbox[3] - y) * yratio)
                pixels.append((px,py))

            px = int(iwidth - ((r.bbox[2] - x) * xratio))
            py = int((r.bbox[3] - y) * yratio)
            pixels.append((px,py))

            draw.polygon(pixels, fill="rgb(" + str(classes[record[1]][0]) + ", " + str(classes[record[1]][1]) + ", " + str(classes[record[1]][2]) + ")")

    img.save(outputFolder + "/" + name + ".png")  
    
    # poly = Polygon(points)        
    # state_polygons[state] = poly
    
       
    


def createPNGfromSHP(shapefile_name, outputFolder):
    filename = basename(shapefile_name)    
    name, file_extension = os.path.splitext(filename)
    
    r = shapefile.Reader(shapefile_name)

    # Determine bounding box x and y distances and then calculate an xyratio
    # that can be used to determine the size of the generated PNG file. A xyratio
    # of greater than one means that PNG is to be a landscape type image whereas
    # an xyratio of less than one means the PNG is to be a portrait type image.
    xdist = r.bbox[2] - r.bbox[0]
    ydist = r.bbox[3] - r.bbox[1]
    xyratio = xdist/ydist
    image_max_dimension = 400 # Change this to desired max dimension of generated PNG

    if (xyratio >= 1):
        iwidth  = image_max_dimension
        iheight = int(image_max_dimension/xyratio)
    else:
        iwidth  = int(image_max_dimension/xyratio)
        iheight = image_max_dimension

    # Iterate through all the shapes within the shapefile and draw polyline
    # representations of them onto the PNGCanvas before saving the resultant canvas
    # as a PNG file
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []

    c = pngcanvas.PNGCanvas(iwidth, iheight)
    # c.color = colors['background']
    c.color = [0, 255, 0, 0]

    for shape in r.shapes():                
        for x,y in shape.points:
            px = int(iwidth - ((r.bbox[2] - x) * xratio))
            py = int((r.bbox[3] - y) * yratio)
            pixels.append([px,py])
        c.polyline(pixels)
        pixels = []

    f = file(outputFolder + "/" + name + ".png", "wb")
    f.write(c.dump())
    f.close()
    

def resterizeShp(shapefile_name, outputFolder):
    filename = basename(shapefile_name)    
    name, file_extension = os.path.splitext(filename)

    NoData_value = -9999
    x_res = 0.03333378 # assuming these are the cell sizes
    y_res = 0.01666641 # change as appropriate
    pixel_size = 1

    shp_object = ogr.Open(shapefile_name)

    

    _layer = shp_object.GetLayer()

    for feature in _layer:
        print feature.GetField("class")
    
    x_min, x_max, y_min, y_max = _layer.GetExtent()

    cols = int((x_max - x_min)/x_res)
    rows = int((y_max - y_min)/y_res)

    _raster = gdal.GetDriverByName('GTiff').Create(outputFolder + "/" + name + ".tif", cols, rows, 1, gdal.GDT_Byte)
    _raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    _band = _raster.GetRasterBand(1)
    _band.SetNoDataValue(NoData_value)

    # print(">> Rasterising shapefile...")
    # gdal.RasterizeLayer(_raster, [1], _layer, burn_values=[0])


    # _output = gdal.GetDriverByName(gdalformat).Create(output, Image.RasterXSize, Image.RasterYSize, 1, datatype, options=['COMPRESS=DEFLATE'])
    # _output.SetProjection(Image.GetProjectionRef())
    # _output.SetGeoTransform(Image.GetGeoTransform()) 
    
    # Band = _output.GetRasterBand(1)

    # Band.SetNoDataValue(0)
    # gdal.RasterizeLayer(_output, [1], _layer, burn_values=1)

    # Band = None
    # Output = None
    # Image = None
    # Shapefile = None

    # subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + output + " 2 4 8 16 32 64", shell=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Rasterize vector, saving colors according to attribute features')       
    parser.add_argument('-shapefile', action="store", dest='strShapefile', help='Shapefile path')
    parser.add_argument('-tile_width', action="store", dest='width', type=int, help='Tile width')
    parser.add_argument('-tile_height', action="store", dest='height', type=int, help='Tile height')
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Folder to store the results')

    result = parser.parse_args()

    # resterizeShp(result.strShapefile, result.outputFolder)
    # createPNGfromSHP(result.strShapefile, result.outputFolder)
    createPNGfromSHP2(result.strShapefile, result.outputFolder, result.width, result.height)

