#!/bin/bash

if [ $# != 4 ] && [ $# != 1 ]
then
    echo "Usage: sh run_distribute_train.sh [CONFIG_PATH] [DATA_PATH] [HYP_PATH] [RANK_TABLE_FILE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 1 ]
then
  RANK_TABLE_FILE=$(get_real_path $1)
  CONFIG_PATH=$"./config/network/yolov3.yaml"
  DATA_PATH=$"./config/data/coco.yaml"
  HYP_PATH=$"./config/data/hyp.scratch.yaml"
fi

if [ $# == 4 ]
then
  CONFIG_PATH=$(get_real_path $1)
  DATA_PATH=$(get_real_path $2)
  HYP_PATH=$(get_real_path $3)
  RANK_TABLE_FILE=$(get_real_path $4)
fi

echo $CONFIG_PATH
echo $DATA_PATH
echo $HYP_PATH
echo $RANK_TABLE_FILE

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end

    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp -r ../config ./train_parallel$i
    cp -r ../network ./train_parallel$i
    cp -r ../utils ./train_parallel$i
    cp -r ../yolov3_backbone.ckpt ./train_parallel
    mkdir ./train_parallel$i/scripts
    cp -r ../scripts/*.sh ./train_parallel$i/scripts/
    cd ./train_parallel || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python train.py \
      --cfg=$CONFIG_PATH \
      --data=$DATA_PATH \
      --hyp=$HYP_PATH \
      --device_target=Ascend \
      --is_distributed=True \
      --epochs=300 \
      --batch-size=16 > log.txt 2>&1 &
cd ..
