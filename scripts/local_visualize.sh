#!/bin/bash

set -e
CONFIG=./projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py
RESULT=work_dirs/results.pkl
VIS_ROOT=vis
VIS_DIR=$VIS_ROOT/vis_results
COMBINED_DIR=$VIS_ROOT/vis_combined
VIDEO_DIR=$VIS_ROOT/videos
VIDEO_NAME=bevformer.mp4
MAX_SHOW_NUM=0
VIS_SCORE_THRESHOLD=0.35
SHOW_GT_CAM=0

rm -rf "$VIS_ROOT"
mkdir -p "$VIS_DIR" "$COMBINED_DIR" "$VIDEO_DIR"

extra_args=()
if [[ "$SHOW_GT_CAM" == "1" ]]; then
	extra_args+=(--show-gt-cam)
fi

python scripts/local_visualize.py \
	--config "$CONFIG" \
	--result "$RESULT" \
	--vis-root "$VIS_ROOT" \
	--max-show-num "$MAX_SHOW_NUM" \
	--vis-score-threshold "$VIS_SCORE_THRESHOLD" \
	"${extra_args[@]}"
