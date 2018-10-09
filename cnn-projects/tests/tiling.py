import os, sys
import gdal
import gdal2tiles
import shapefile
import osr
import geopandas as gp

from shapely.geometry import Polygon
from os.path import basename

# USAGE: python tiling.py PATH_TO_IMAGE PATH_TO_SHAPEFILE WIDTH_OF_THE_TILE HEIGHT_OF_THE_TILE SAVE_DESTINATION
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
    # proj = osr.SpatialReference(wkt=ds.GetProjection()) 

    if ds is None:
        print 'Could not open image file'
        sys.exit(1)

    rows = ds.RasterXSize
    cols = ds.RasterYSize
    # bands = ds.RasterCount
        
    #for i in xrange(1, bands + 1):
        # band = ds.GetRasterBand(i)
        # min.append(band.GetMinimum())
        # max.append(band.GetMaximum())

    # for i in range(0, rows, width):
    #     for j in range(0, cols, height):            
    #         # gdal2tiles.generate_tiles(infile, output_folder, np_processes=2, zoom='10-15') #maptiler alternative
    #         com_string = "gdal_translate -of GTIFF -co TILED=YES -ot UInt16 -srcwin " + str(i) + ", " + str(j) + ", " + str(width) + ", " + str(height) + " " + infile + " " + output_folder + name + "_" + str(i) + "_" + str(j) + file_extension            
    #         os.system(com_string)

    for i in range(0, rows, width):
        for j in range(0, cols, height):            
            tile = gdal.Open(output_folder + name + "_" + str(i) + "_" + str(j) + file_extension)
            
            gt = tile.GetGeoTransform()              
            tileWidth = tile.RasterYSize
            tileHeight = tile.RasterXSize
                        
            minx = gt[0]
            maxy = gt[3]
            miny = gt[3] + tileHeight * gt[5]  
            maxx = gt[0] + tileWidth * gt[1] 
            
            # print(str(minx) + " " + str(maxy) + " " + str(maxx) + " " + str(miny))
            
            # bounds = Polygon( [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)] )
            # baseshp = gp.read_file(strShapefile)
            # baseshp['geometry'] = baseshp['geometry'].intersection(bounds)
            # baseshp[baseshp.geometry].to_file(output_folder + name + "_" + str(i) + "_" + str(j) + ".shp", driver='ESRI Shapefile')
            # baseshp[baseshp.geometry].to_file(output_folder + name + "_" + str(i) + "_" + str(j) + ".shp", driver='ESRI Shapefile')
               
            com_string = "ogr2ogr -f \"ESRI Shapefile\" " + output_folder + name + "_" + str(i) + "_" + str(j) + ".shp " + strShapefile + " -clipsrc " + str(minx) + " " + str(miny) + " " + str(maxx) + " " + str(maxy)
            os.system(com_string)    

            
            