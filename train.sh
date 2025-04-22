base_dir="./nfa_vit_train"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
train.py \
    --model NFA_ViT \
    --world_size 1 \
    --find_unused_parameters \
    --if_not_amp\
    --batch_size 64 \
    --test_batch_size 64 \
    --data_path  "clpbc/brgen_train"\
    --epochs 30 \
    --lr 5e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --sparse_ratio 0.25 \
    --sparse_rate 2 \
    --weight_decay 0.005 \
    --test_data_path "clpbc/brgen_test" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 42 \
    --test_period 5 \
    --num_workers 12 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log

