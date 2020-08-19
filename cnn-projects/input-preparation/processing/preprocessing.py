import os
import glob
import logging
import settings
import warnings
import numpy as np

from osgeo import gdal
from osgeo.gdalconst import *


class Processing:
    """
    """

    def __init__(self):
        pass

    def stacking(self, safe_folder, output_path, file_pattern):
        """
        :param safe_folder:
        :param output_path:
        :param file_pattern:
        :return:
        usage: stacking('/data/ITEM_IMAGE.SAFE/', '/data/stack/', "_10m.jp2")
        """
        if os.path.isdir(safe_folder):
            safe_item = safe_folder.split(".")[0]
            list_images = glob.glob(safe_folder + "/**/*" + file_pattern, recursive=True)

            if len(list_images) != 0:
                if len(list_images) == len(settings.PARAMS['S2']['bands']):
                    logging.info(">> Stacking scene {}...".format(safe_folder))

                    output_path_aux = os.path.join(output_path, safe_item + "_" + "stk.tif")

                    command = "gdal_merge.py -of gtiff -ot float32 -co COMPRESS=NONE -co BIGTIFF=IF_NEEDED " \
                              "-separate -n 0 -a_nodata 0 " + " -o " + output_path_aux + " " + " ".join(list_images)

                    os.system(command)
            else:
                logging.warning(
                    ">>>> The path {} is either empty, no .zip or SAFE formats available!".format(safe_folder))

    def statistic(self, procedure, stack_folder, output_path, file_pattern):
        """
        :param procedure: mean, median, mode, var, std
        :param stack_folder:
        :param output_path:
        :param file_pattern:
        :return:
        usage: statistic('mean', '/data/stack/', '/data/stats/', "*.tif")
        """
        if os.path.isdir(stack_folder):
            list_images = glob.glob(stack_folder + file_pattern, recursive=True)

            if len(list_images) != 0:
                logging.info(">> Calculating {} image...".format(procedure))

                output_path_aux = os.path.join(output_path, "image_mean.tif")

                for i, item in enumerate(list_images):
                    image_path = os.path.join(stack_folder, item)

                    in_ds = gdal.Open(image_path)
                    driver = gdal.GetDriverByName('GTiff')
                    rows = in_ds.RasterYSize
                    cols = in_ds.RasterXSize
                    array = in_ds.ReadAsArray()
                    array = np.expand_dims(array, 2)

                    if i == 0:
                        layers = array
                    else:
                        layers = np.concatenate((layers, array), axis=2)

                    # logging.info(">>>> Separating stacked layers...")
                    # for i in range(1, in_ds.RasterCount + 1):
                    #     arr = in_ds.GetRasterBand(i).ReadAsArray()
                    #     arr = np.expand_dims(arr, 2)
                    #     arr[arr == 0] = np.nan
                    #     if i == 1:
                    #         layers = arr
                    #     else:
                    #         layers = np.concatenate((layers, arr), axis=2)

                logging.info(">>>> Calculating mean...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_raster = np.nanmean(layers, axis=2)

                out_ds = driver.Create(output_path_aux, cols, rows, 3, GDT_Float32)
                out_ds.SetGeoTransform(in_ds.GetGeoTransform())
                out_ds.SetProjection(in_ds.GetProjection())

                if out_ds is None:
                    logging.warning(">>>>>> Could not create output mean image {}.".format(output_path_aux))
                    return

                logging.info(">>>> Creating output {}...".format(output_path_aux))
                out_band = out_ds.GetRasterBand(1)
                out_band.WriteArray(mean_raster)
                out_band.FlushCache()
                out_band = None

            else:
                logging.warning(">>>> The path {} is either empty, no .zip or SAFE formats available!".format(stack_folder))
