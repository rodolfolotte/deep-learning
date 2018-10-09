#!/bin/sh
for entry in $1; do
	if [ ! -d "$entry"/segmentation/ ]; then
 		 python segment.py --input_folder $2 --output_folder "$entry"/segmentation/ --logdir "$entry"
	fi
done
