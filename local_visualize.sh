#!/bin/bash

python tools/misc/visualize_results.py \
	./projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py \
	--result work_dirs/results.pkl \
	--show-dir vis \
	--max-show-num 5
