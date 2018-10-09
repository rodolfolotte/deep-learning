import os, sys
import argparse
import gdal
import shapefile
import pngcanvas
import osgeo.osr as osr, ogr

from os.path import basename
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.geometry import Point

# USAGE: python shp2raster.py -shapefile SHAPEFILE_FOLDER -output_foler OUTPUT_SHP_RASTER -tile_width INT_SIZE_OF_THE_RASTER_IN_COLUMNS -tile_height INT_SIZE_OF_THE_RASTER_IN_ROWS
# EXAMPLE: python shp2raster.py -shapefile /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile-shp2/ -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile2/ -tile_width 400 -tile_height 400

classes = {    
    "def": [255,255,0],
    "water": [102,204,255],
    "cloud": [255,255,255],
    "shadow": [128,128,128]
}

def sliceArray(array, positions):
    new_arrays=[]     
    positions.append(len(array)-1) 
    for i in range(len(positions)-1):        
        new_arrays.append(array[positions[i]:positions[i+1]])        
    
    return new_arrays


# SOURCE: https://github.com/GeospatialPython/geospatialpython/blob/master/shp2img.py
def createPNGfromSHP(shapefileFolder, outputFolder, iwidth, iheight):
    valid_files = [".shp"]

    for f in os.listdir(shapefileFolder):        
        name, file_extension = os.path.splitext(f)

        if file_extension.lower() not in valid_files:            
            continue

        r = shapefile.Reader(shapefileFolder + "/" + f)
        if not r:
            print('>> Error: could not open the shapefile')

        xdist = r.bbox[2] - r.bbox[0]
        ydist = r.bbox[3] - r.bbox[1]    
        xratio = iwidth / xdist
        yratio = iheight / ydist

        shapes = r.shapes()            
        records = r.records()
                        
        img = Image.new("RGB", (iwidth, iheight), "black")
        draw = ImageDraw.Draw(img)

        for i, record in enumerate(records):             
            if record[1] in classes:                                              
                parts = shapes[i].parts                 
                pixels = []                
                for x,y in shapes[i].points: 
                    px = int(iwidth - ((r.bbox[2] - x) * xratio))
                    py = int((r.bbox[3] - y) * yratio)
                    pixels.append((px,py))                                              
                                
                polygons_parts=[]                                     
                if(len(parts) > 1):                                                          
                    polygons_parts = sliceArray(pixels, parts)
                    for k in range(len(polygons_parts)):                    
                        draw.polygon(polygons_parts[k], outline=None, fill="rgb(" + str(classes[record[1]][0]) + ", " + str(classes[record[1]][1]) + ", " + str(classes[record[1]][2]) + ")")
                else:                    
                    draw.polygon(pixels, outline=None, fill="rgb(" + str(classes[record[1]][0]) + ", " + str(classes[record[1]][1]) + ", " + str(classes[record[1]][2]) + ")")

        img.save(outputFolder + "/" + name + ".png")
        print(">> Raster (png) respect to vector " + f + " save successfully!")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Rasterize vector, saving colors according to attribute features')       
    parser.add_argument('-shapefile', action="store", dest='shpFolder', help='Shapefile folder path')
    parser.add_argument('-tile_width', action="store", dest='width', type=int, help='Tile width')
    parser.add_argument('-tile_height', action="store", dest='height', type=int, help='Tile height')
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Folder to store the results')

    result = parser.parse_args()
    
    createPNGfromSHP(result.shpFolder, result.outputFolder, result.width, result.height)

