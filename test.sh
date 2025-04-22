base_dir="./nfa_vit_test"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
test.py \
    --model NFA_ViT \
    --world_size 1 \
    --test_data_json "clpbc/brgen_test" \
    --checkpoint_path "nfa_vit_train" \
    --test_batch_size 64 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log