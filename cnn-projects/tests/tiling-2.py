import os, sys
import gdal
import osgeo.osr as osr, ogr
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

    # for i in range(0, rows, width):
    #     for j in range(0, cols, height):            
    #         # gdal2tiles.generate_tiles(infile, output_folder, np_processes=2, zoom='10-15') #maptiler alternative
    #         com_string = "gdal_translate -of GTIFF -co TILED=YES -ot UInt16 -srcwin " + str(i) + ", " + str(j) + ", " + str(width) + ", " + str(height) + " " + infile + " " + output_folder + name + "_" + str(i) + "_" + str(j) + file_extension            
    #         os.system(com_string)

    for i in range(0, rows, width):
        for j in range(0, cols, height):            
            tile = gdal.Open(output_folder + name + "_" + str(i) + "_" + str(j) + file_extension)
            
            # tile_projection = tile.GetProjectionRef()    
            # driver = ogr.GetDriverByName('ESRI Shapefile')
            # ds = driver.CreateDataSource(shapefile)
            # srs = osr.SpatialReference(wkt=tile_projection)     
            # srs.ImportFromEPSG(4326)
            # gt = tile.GetGeoTransform() 

            source = osr.SpatialReference()
            source.ImportFromWkt(tile.GetProjection())

            target = osr.SpatialReference()
            target.ImportFromEPSG(4326)
            
            transform = osr.CoordinateTransformation(source, target)

            ulx, xres, xskew, uly, yskew, yres  = tile.GetGeoTransform()
            lrx = ulx + (tile.RasterXSize * xres)
            lry = uly + (tile.RasterYSize * yres)
            
            transform.TransformPoint(ulx, uly)
            print(ulx, uly, lrx, lry)

            # com_string = "ogr2ogr -f \"ESRI Shapefile\" " + output_folder + name + "_" + str(i) + "_" + str(j) + ".shp " + strShapefile + " -clipsrc " + str(geo_ext[0]) + " " + str(geo_ext[1]) + " " + str(geo_ext[2]) + " " + str(geo_ext[3])
            # os.system(com_string)    

            
            