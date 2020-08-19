import sys
import logging
import argparse
import settings

from coloredlogs import ColoredFormatter
from tile import tiling
from processing import preprocessing


def main(arguments):
    """
    :param arguments:
    :return :
    """
    if arguments.procedure is not None:
        if arguments.procedure == 'tiling_raster':
            if (arguments.image_folder is not None) and (arguments.output_folder is not None) \
                    and (arguments.width is not None) and (arguments.height is not None):
                tiling.Tiling().tiling_raster(arguments.image_folder, arguments.output_folder,
                                              arguments.width, arguments.height)
            else:
                logging.error(">> One of arguments (image_folder, output_folder, wigth, height) are incorrect or "
                              "empty. Try it again!")
                raise RuntimeError
        elif arguments.procedure == 'tiling_vector':
            if (arguments.image_folder is not None) and (arguments.shapefile_reference is not None) and \
                    (arguments.output_folder is not None):
                tiling.Tiling().tiling_vector(arguments.image_folder, arguments.shapefile_reference,
                                              arguments.output_folder)
            else:
                logging.error(">> One of arguments (image_folder, shapefile_reference, output_folder) are incorrect or "
                              "empty. Try it again!")
                raise RuntimeError
        elif arguments.procedure == 'shp2png':
            if (arguments.shapefile_folder is not None) and (arguments.output_folder is not None) and \
                    (arguments.width is not None) and (arguments.height is not None):
                tiling.Tiling().shp2png(arguments.shapefile_folder, arguments.output_folder,
                                        arguments.width, arguments.height, settings.CLASSES)
            else:
                logging.error(">> One of arguments (image_folder, shapefile_reference, output_folder) are incorrect or "
                              "empty. Try it again!")
                raise RuntimeError
        elif arguments.procedure == 'stack':
            if (arguments.image_folder is not None) and (arguments.output_folder is not None):
                preprocessing.Processing().stacking(arguments.image_folder, arguments.output_folder, '_10m.jp2')
            else:
                logging.error(">> One of arguments (image_folder, output_folder) are incorrect or "
                              "empty. Try it again!")
                raise RuntimeError
        elif arguments.procedure == 'stat':
            if (arguments.image_folder is not None) and (arguments.output_folder is not None):
                preprocessing.Processing().statistic('mean', arguments.image_folder, arguments.output_folder, '*.tif')
            else:
                logging.error(">> One of arguments (image_folder, output_folder) are incorrect or "
                              "empty. Try it again!")
                raise RuntimeError
        else:
            logging.error(">> Procedure option not found. Try it again!")
            raise RuntimeError


if __name__ == '__main__':
    """ Command-line routine that through two folders: original images and ground-truth, build validation, testing 
    and training .txt files. Each line in these files represents pairs of paths: original raster SPACE raster reference. 
    In the training file, all the pairs of images to be used in the training of the neural model are listed, while in 
    the validation file, images are placed for validation according to the percentage specified 
    USAGE:
        python main.py -procedure tiling_raster 
                       -image_folder /data/bioverse/images/cachoeira-porteira/pan-sharp-mosaic/ 
                       -output_folder /data/bioverse/images/cachoeira-porteira/pan-sharp-mosaic/tiles/
                       -tile_width 128 -tile_height 128 
                       -verbose True
        python main.py -procedure tiling_vector
                       -image_folder /data/bioverse/images/cachoeira-porteira/pan-sharp-mosaic/tiles/ 
                       -output_folder /data/bioverse/images/cachoeira-porteira/pan-sharp-mosaic/annotation/
                       -shapefile_reference /data/bioverse/images/cachoeira-porteira/ground_reference/*.shp
                       -verbose True     
        python main.py -procedure shp2png
                       -shapefile_folder /data/prodes/dl/deforestation/ground-truth/tiles/vector/
                       -output_folder /data/prodes/dl/deforestation/ground-truth/tiles/png/
                       -tile_width 256 -tile_height 256
                       -verbose True
        python main.py -procedure stack
                       -image_folder /data/prodes/dl/deforestation/ground-truth/tiles/vector/
                       -output_folder /data/prodes/dl/deforestation/ground-truth/tiles/png/                       
                       -verbose True
        python main.py -procedure stat
                       -image_folder /data/lotte/training-dataset/S2/2019-01-01_2019-06-30/aoi_4/stack/
                       -output_folder /data/lotte/training-dataset/S2/2019-01-01_2019-06-30/aoi_4/mean/             
                       -verbose True
    """
    parser = argparse.ArgumentParser(description='Prepare input files for supervised neural network procedures')

    parser.add_argument('-procedure', action="store", dest='procedure', help='Procedure to be performed. Options: '
                                                                             'tiling_vector, tiling_raster')
    parser.add_argument('-image_folder', action="store", dest='image_folder', help='Images folder')
    parser.add_argument('-output_folder', action="store", dest='output_folder', help='Folder to store the input files')
    parser.add_argument('-shapefile_reference', action="store", dest='shapefile_reference',
                        help='ESRI Shapefile to be used as reference to generate the respective annotation for image '
                             'tiles. The image_folder argument, in this case, has to be the image tiles folder')
    parser.add_argument('-shapefile_folder', action="store", dest='shapefile_folder',
                        help='')
    parser.add_argument('-tile_width', action="store", dest='width', type=int, help='Tile width')
    parser.add_argument('-tile_height', action="store", dest='height', type=int, help='Tile height')
    parser.add_argument('-verbose', action="store", dest='verbose', help='Print log of processing')

    args = parser.parse_args()

    if eval(args.verbose):
        log = logging.getLogger('')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cf = ColoredFormatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ")
        ch.setFormatter(cf)
        log.addHandler(ch)

        fh = logging.FileHandler('logging.log')
        fh.setLevel(logging.INFO)
        ff = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ",
                               datefmt='%Y.%m.%d %H:%M:%S')
        fh.setFormatter(ff)
        log.addHandler(fh)

        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(args)
