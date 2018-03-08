#!/bin/sh
for entry in $1
do
	for entry in $1
	do
        	python segment.py --input_folder $2 --output_folder "$entry"/segmentation/ --logdir "$entry"
	done
done

