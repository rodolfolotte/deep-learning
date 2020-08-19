#! /bin/bash
#python tiling.py -choose 0 -image_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/not-tiled/ -tile_width 400 -tile_height 400 -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/tile/

# python tiling.py -choose 1 -image_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/original/tile/ -shapefile_reference /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/not-tiled/ -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile-shp/ -reproject 1 

python shp2raster.py -shapefile /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile-shp/ -output_folder /home/lotte/Bit/lotte/personal/sccon/deforestation-planet/data/annotation/tile/ -tile_width 400 -tile_height 400
