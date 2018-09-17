import os, sys
import gdal

from os.path import basename

# USAGE: python tilingImage.py PATH_TO_IMAGE WIDTH_OF_THE_TILE HEIGHT_OF_THE_TILE SAVE_DESTINATION
if __name__=='__main__':
    infile = sys.argv[1]    
    width, height = int(sys.argv[2]), int(sys.argv[3])
    output_folder = sys.argv[4]

    filename = basename(infile)    
    name, file_extension = os.path.splitext(filename)
    
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