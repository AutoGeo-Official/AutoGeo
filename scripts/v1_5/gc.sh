for lr in 6e-5 ; do
    for epoch in 2; do
        
        # 调整data_path和image_folder在不同的数据集上进行训练
        torchrun --nproc_per_node=8 llava/train/train_mem.py \
            --lora_enable True --lora_r 128 --lora_alpha 256 \
            --deepspeed ./scripts/zero3.json \
            --tune_mm_mlp_adapter True \
            --tune_vision False \
            --model_name_or_path liuhaotian/llava-v1.5-7b \
            --data_path ./data/alphageometry/dataset_reinforce_100k/data.json \
            --image_folder ./data/alphageometry/dataset_reinforce_100k/ \
            --bf16 True \
            --output_dir ./checkpoints/llava-100k \
            --num_train_epochs $epoch \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 200 \
            --save_total_limit 10 \
            --learning_rate $lr \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --model_max_length 512 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb
        
    done
done

# 生成结果
for lr in 6e-5 ; do
    for epoch in 2; do
        echo lr ${lr} epoch ${epoch} 
        torchrun llava/eval/model_vqa.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/llava-100k/non_lora_trainables.bin \
        --lora_ckpt ./checkpoints/llava-100k/  \
        --question-file data/questions/question.jsonl \
        --image-folder data/alphageometry/dataset_test/img/ \
        --answers-file output/answer-file-7b-mmpvision-lr${lr}-epoch${epoch}-100k.jsonl 
    done
done


# 计算生成结果的分数
for lr in 6e-5 ; do
    for epoch in 2; do
        echo lr ${lr} epoch ${epoch} 
        python -m llava.eval.eval_score \
            --annotation-file ./data/alphageometry/dataset_test/data.json \
            --result-file output/answer-file-7b-mmpvision-lr${lr}-epoch${epoch}-100k.jsonl
    done
done






