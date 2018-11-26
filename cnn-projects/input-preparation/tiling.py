#!/usr/bin/env python

"""
Command-line routine to tile images and shapefiles according to desired width and heights
"""

__author__ = 'Rodolfo G. Lotte'
__copyright__ = 'Copyright 2018, Rodolfo G. Lotte'
__credits__ = ['Rodolfo G. Lotte']
__example_raster__ = 'python tiling.py -choose 0 -image_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/not-tiled/ -tile_width 200 -tile_height 200 -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/tile/'
__example_vector__ = 'python tiling.py -choose 1 -image_tiles /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/tile/ -shapefile /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/not-tiled/20170707_131443_1044.shp -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile/ -reproject 1'
__license__ = 'MIT'
__email__ = 'rodolfo.lotte@gmail.com'

import argparse
import os
import sys
import gdal
import logging
import geopandas as gp
import osgeo.osr as osr

from os.path import basename
from shapely.geometry import Polygon

log = logging.getLogger('')
log.setLevel(logging.INFO)
format = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ", datefmt='%Y.%m.%d %H:%M:%S')

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)

fh = logging.handlers.RotatingFileHandler(filename='inputs.log', maxBytes=(1048576*5), backupCount=7)
fh.setFormatter(format)

log.addHandler(ch)
log.addHandler(fh)

# https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
def GetExtent(gt, cols, rows):
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])            
        yarr.reverse()
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


def tilingImage(folder, width, height, outputFolder):    
    valid_images = [".jpg",".gif",".png",".tga",".tif"]

    logging.info("Tiling images in folder " + folder + "...")
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1]

        if ext.lower() not in valid_images:            
            continue

        logging.info(">> Image " + f + "...")
        complete_path = os.path.join(folder, f)
        filename = basename(complete_path)        
        name, file_extension = os.path.splitext(filename)

        if(not os.path.isfile(complete_path)):
            logging.info( ">>>> Tile " + f + " does not exist. Check it and try again!")
            continue
        
        ds = gdal.Open(complete_path)   
        
        if ds is None:
            logging.info('>>>> Could not open image file!')
            sys.exit(1)

        rows = ds.RasterXSize
        cols = ds.RasterYSize

        gdal.UseExceptions()
        for i in range(0, rows, width):
            for j in range(0, cols, height):
               try:
                   # https://www.gdal.org/gdal_translate.html
                   gdal.Translate(outputFolder + name + "_" + str(i) + "_" + str(j) + file_extension, ds, format='GTIFF', srcWin=[i, j, width, height], outputType=gdal.GDT_UInt16, options=['-eco'])

               except RuntimeError as error:
                   logging.info(">>>> Completely outside the image. " + error + " Not saved!")
                   pass



def tilingShape(imagesFolder, shpReferenceFolder, outputFolder, reproject):
    valid_vectors = [".shp"]
    valid_images = [".jpg",".gif",".png",".tga",".tif"]

    logging.info("Tiling vectors in folder " + shpReferenceFolder + "...")
    for f in os.listdir(shpReferenceFolder):
        ext = os.path.splitext(f)[1]
        
        if ext.lower() not in valid_vectors:            
            continue
        
        complete_shapefile_path = os.path.join(shpReferenceFolder, f)

        if(not os.path.isfile(complete_shapefile_path)):
            logging.info( ">> Vector tile " + f + " does not exist. Check it and try again!")
            continue
        else:             
            # shapefile_name, shapefile_extension = os.path.splitext(f)

            logging.info(">> Tiling vector " + f + " respecting to the tiles extends in " + imagesFolder)
            for im in os.listdir(imagesFolder):
                ext2 = os.path.splitext(im)[1]
                
                if ext2.lower() not in valid_images:
                    continue

                complete_path = os.path.join(imagesFolder, im)
                filename = basename(complete_path)        
                name, file_extension = os.path.splitext(filename)
                                
                tile = gdal.Open(complete_path)

                # tile_projection = tile.GetProjectionRef()
                gt = tile.GetGeoTransform()                         
                cols_tile = tile.RasterXSize
                rows_tile = tile.RasterYSize
                ext = GetExtent(gt, cols_tile, rows_tile)

                if(reproject==0):                             
                    srs_tile = osr.SpatialReference()
                    srs_tile.ImportFromWkt(tile.GetProjection())
                    tgt_srs = srs_tile.CloneGeogCS()
                    ext = ReprojectCoords(ext, srs_tile, tgt_srs)            

                bounds = Polygon(ext)
                baseshp = gp.read_file(complete_shapefile_path)  
                crs = baseshp.crs                 
                  
                ids = []
                classes = []
                polygons_intersecting = []
                for i in range(len(baseshp)):                                               
                    if(baseshp['geometry'][i].intersection(bounds).is_empty == False):       
                        ids.append(i)             
                        classes.append(baseshp['class'][i])
                        polygons_intersecting.append(baseshp['geometry'][i].intersection(bounds))
                            
                gdf = gp.GeoDataFrame()
                gdf.crs = crs                        
                if(len(polygons_intersecting) != 0):
                    gdf['id'] = ids
                    gdf['class'] = classes       
                    gdf['geometry'] = polygons_intersecting
                    gdf.to_file(outputFolder + name + ".shp", driver='ESRI Shapefile')
                

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Tiling images and shapefiles')
    parser.add_argument('-choose', action="store", dest='choose', type=int, help='0 for image, 1 for shapefile')    

    parser.add_argument('-image_folder', action="store", dest='imageFolder', help='Images folder to be tiled')        
    parser.add_argument('-tile_width', action="store", dest='width', type=int, help='Tile width')
    parser.add_argument('-tile_height', action="store", dest='height', type=int, help='Tile height')        
    parser.add_argument('-image_tiles', action="store", dest='imageTiles', help='Image tiles to be used as references')
    parser.add_argument('-shapefile_reference', action="store", dest='shapefileReferenceFolder', help='Shapefiles folder, the folder with shapefiles references in order to be tiled')    
    parser.add_argument('-reproject', action="store", dest='reproject', type=int, help='In some case, the reprojection is needed')    
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Folder to store the results')

    result = parser.parse_args()

    if(result.choose == 0):
        tilingImage(result.imageFolder, result.width, result.height, result.outputFolder)
    elif(result.choose == 1):
        tilingShape(result.imageTiles, result.shapefileReferenceFolder, result.outputFolder, result.reproject)