#!/bin/bash

if [ $# != 2 ] && [ $# != 5 ]
then
    echo "Usage: bash run_standalone_test_ascend.sh [WEIGHTS] [DEVICE_ID]"
    echo "OR"
    echo "Usage: bash run_standalone_test_ascend.sh [WEIGHTS] [DEVICE_ID] [CONFIG_PATH] [DATA_PATH] [HYP_PATH]"
exit 1
fi

WEIGHTS=$1
export DEVICE_ID=$2
export RANK_ID=$2

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 2 ]
then
  CONFIG_PATH=$"./yolov3.yaml"
  DATA_PATH=$"./coco.yaml"
  HYP_PATH=$"./hyp.scratch.yaml"
fi

if [ $# == 5 ]
then
  CONFIG_PATH=$(get_real_path $3)
  DATA_PATH=$(get_real_path $4)
  HYP_PATH=$(get_real_path $5)
fi

echo $CONFIG_PATH
echo $DATA_PATH
echo $HYP_PATH


export DEVICE_NUM=1
export RANK_SIZE=1
rm -rf ./test_standalone$2
mkdir ./test_standalone$2
cp ./*.py ./test_standalone$2
cp ./coco.yaml ./test_standalone$2
cp ./yolov3.yaml ./test_standalone$2
cp ./hyp.scratch.yaml ./test_standalone$2
cp ./EMA_yolov3_300.ckpt ./test_standalone$2
mkdir ./test_standalone$2/scripts
cp -r ./*.sh ./test_standalone$2/scripts/
cd ./test_standalone$2 || exit
env > env.log
python val.py \
  --weights=$WEIGHTS \
  --cfg=$CONFIG_PATH \
  --data=$DATA_PATH \
  --hyp=$HYP_PATH \
  --device_target=Ascend \
  --img-size=640 \
  --conf=0.001 \
  --iou=0.6 \
  --batch-size=32 > log.txt 2>&1 &
cd ..