import os, sys
import gdal
import osgeo.osr as osr, ogr
import numpy as np
import geopandas as gp

from shapely.geometry import Polygon
from os.path import basename

# USAGE: python tiling.py PATH_TO_IMAGE PATH_TO_SHAPEFILE WIDTH_OF_THE_TILE HEIGHT_OF_THE_TILE SAVE_DESTINATION

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


if __name__=='__main__':
    infile = sys.argv[1]
    strShapefile = sys.argv[2]
    width, height = int(sys.argv[3]), int(sys.argv[4])
    output_folder = sys.argv[5]

    filename = basename(sys.argv[1])
    shapeDir = os.path.dirname(sys.argv[2])
    shapeFilename = basename(sys.argv[2])
    name, file_extension = os.path.splitext(filename)
    shapename, shape_extension = os.path.splitext(shapeFilename)

    ds = gdal.Open(infile)   
    
    if ds is None:
        print 'Could not open image file'
        sys.exit(1)

    rows = ds.RasterXSize
    cols = ds.RasterYSize

    for i in range(0, rows, width):
        for j in range(0, cols, height):                        
            com_string = "gdal_translate -epo -eco -of GTIFF -ot UInt16 -srcwin " + str(i) + ", " + str(j) + ", " + str(width) + ", " + str(height) + " " + infile + " " + output_folder + name + "_" + str(i) + "_" + str(j) + file_extension            
            os.system(com_string)

    # for i in range(0, rows, width):
    #     for j in range(0, cols, height):     
    #         tile_name = output_folder + name + "_" + str(i) + "_" + str(j) 

    #         if(not os.path.isfile(tile_name + file_extension)):
    #             print "Tile " + tile_name + " does not exist. Check it and try again!"
    #             continue
    #         else:
    #             tile = gdal.Open(tile_name + file_extension)

    #             tile_projection = tile.GetProjectionRef()                
    #             gt = tile.GetGeoTransform()                         
    #             cols_tile = tile.RasterXSize
    #             rows_tile = tile.RasterYSize
    #             ext = GetExtent(gt, cols_tile, rows_tile)

    #             srs_tile = osr.SpatialReference()
    #             srs_tile.ImportFromWkt(tile.GetProjection())
    #             tgt_srs = srs_tile.CloneGeogCS()
    #             geo_ext = ReprojectCoords(ext, srs_tile, tgt_srs)
                                
    #             bounds = Polygon(geo_ext)
    #             baseshp = gp.read_file(strShapefile)                   

    #             if(baseshp['geometry'] is not None):
    #                 print(baseshp['geometry'])
    #                 baseshp['geometry'] = baseshp['geometry'].intersection(bounds)
    #                 baseshp.to_file(tile_name + ".shp", driver='ESRI Shapefile')   
    #                 print("Shapefile refereing to tile " + name + "_" + str(i) + "_" + str(j) + file_extension + " save successfully!")
    #             else:
    #                 print("Geometry refereing to tile " + name + "_" + str(i) + "_" + str(j) + file_extension + " has no information or it is corrupted. Skipped!")

                