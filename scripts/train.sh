#!/usr/bin/env bash

set -x

# set dist args
# SINGLE=1
# nproc_per_node=${ARNOLD_WORKER_GPU}

# if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
#   echo "[single node alone] SINGLE=$SINGLE"
#   nnodes=1
#   node_rank=0
#   nproc_per_node=1
#   master_addr=127.0.0.1
#   master_port=12345
# else
#   MASTER_NODE_ID=0
#   nnodes=${ARNOLD_WORKER_NUM}
#   node_rank=${ARNOLD_ID}
#   master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
#   master_addr=${!master_addr}
#   master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
#   master_port=${!master_port}
#   ports=(`echo $master_port | tr ',' ' '`)
#   master_port=${ports[0]}
# fi

# echo "[nproc_per_node: ${nproc_per_node}]"
# echo "[nnodes: ${nnodes}]"
# echo "[node_rank: ${node_rank}]"
# echo "[master_addr: ${master_addr}]"
# echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0


BED=/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1/exp        # checkpoint 根目录
LOCAL_OUT=./outputs # 本地输出根目录（日志/可视化等）
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# wandb offline
wandb login --relogin 711f941f459be2c398272020e434baaf9bb1b2e7
wandb online  # 将 W&B 切到离线模式（不需要登录，不上传；本地生成 wandb/ 目录）

exp_name=FGSC_030401
bed_path=checkpoints/${exp_name}/
model_path=/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1/models
data_path=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/FGSC/train

video_data_path=''
local_out_path=$LOCAL_OUT/${exp_name}

# rm -rf ${bed_path}
# rm -rf ${local_out_path}

torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=$RANK \
--master_addr=$MASTER_ADDR \
--master_port=23456 \
train.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='RS' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=24e-5 \
--pn 0.06M \
--model=layer12 \
--lbs=32 \
--workers=4 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 3000 \
--Ct5=2048 \
--t5_path=$model_path/google/flan-t5-xl \
--vae_type 16 \
--vae_ckpt=$model_path/FoundationVision/Infinity/infinity_vae_d16.pth \
--wp 0.05 \
--wpe 0.01 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 0 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=0 \
--save_model_iters_freq 100 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.1 \
--noise_apply_layers 5 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128 \
--debug=0 \