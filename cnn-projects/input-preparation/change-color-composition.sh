#! /bin/bash
for file in $1*.tif; do    
	noextension="${file%.*}"
    filename=$(basename "$noextension")
    output=$2$filename'.tif'

    gdal_translate -b 3 -b 2 -b 1 $file $output	
done
