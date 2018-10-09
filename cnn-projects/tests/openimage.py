import tifffile as tiff
from libtiff import TIFF
import cv2

a = tiff.imread("/home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/20170707_131443_1044_400_2000.tif")
# tif = TIFF.open("/home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/20170707_131443_1044_400_2000.tif")

# image = tif.read_image()

if(a.dtype == 'uint8'):
    print("uint8")

if(a.dtype == 'uint16'):
    print("uint16")    
    image = cv2.convertScaleAbs(a)

print(image)
