base_dir="./outputs/train_nfavit"
mkdir -p ${base_dir}

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
./IMDLBenCo/IMDLBenCo/training_scripts/train.py \
    --model NFA_ViT \
    --world_size 1 \
    --batch_size 16 \
    --data_path ./datasets/BRGen_train.json \
    --epochs 200 \
    --lr 2e-4 \
    --min_lr 0 \
    --weight_decay 0.05 \
    --np_pretrain_weights "./model_zoo/nfavit/noiseprint.pth" \
    --seg_b0_pretrain_weights "./model_zoo/nfa_vit/segformer_b0_backbone_weights.pth" \
    --seg_b2_pretrain_weights "./model_zoo/nfa_vit/segformer_b2_backbone_weights.pth" \
    --test_data_path ./datasets/BRGen_test.json \
    --if_resizing \
    --find_unused_parameters \
    --warmup_epochs 4 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 20 \
    --num_workers 12 \
