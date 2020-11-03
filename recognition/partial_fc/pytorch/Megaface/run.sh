#!/usr/bin/env bash

MEGAFACE="/home/yiminglin/insightface/datasets/megaface/"
ln -sf $MEGAFACE ./data
DEVKIT="/home/yiminglin/insightface/datasets/megaface/devkit"
ALGO="partial_fc_original" #ms1mv2
ROOT=$(dirname `which $0`)
echo $ROOT
python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model '../models/19backbone.pth, 19' 

python -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" python -u run_experiment.py "$ROOT/feature_out_clean/megaface" "$ROOT/feature_out_clean/facescrub" _"$ALGO".bin $ROOT/$ALGO -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

