#!/bin/bash

python tools/test.py \
	./projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py \
	./ckpts/epoch_24.pth \
	--eval bbox \
	--out work_dirs/results.pkl
