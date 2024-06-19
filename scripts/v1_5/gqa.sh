
# 在qa数据上训练
for lr in 6e-5 ; do
    for epoch in 3; do  

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=7 llava/train/train_mem_qa.py \
            --lora_enable True --lora_r 256 --lora_alpha 512 \
            --deepspeed ./scripts/zero2.json \
            --tune_mm_mlp_adapter True \
            --tune_vision False \
            --model_name_or_path liuhaotian/llava-v1.5-7b \
            --pretrain_mm_mlp_adapter ./checkpoints/mmpvisionlora-529-lr6e-5-epoch2-100k/non_lora_trainables.bin \
            --lora_ckpt ./checkpoints/mmpvisionlora-529-lr6e-5-epoch2-100k/  \
            --data_path ./playground/data/qa_tuning.json \
            --image_folder ./playground/data/images \
            --bf16 True \
            --output_dir ./checkpoints/qa_tune100k_256_${epoch} \
            --num_train_epochs ${epoch} \
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
            --logging_steps 2 \
            --model_max_length 512 \
            --gradient_checkpointing True \
            --dataloader_num_workers 0 \
            --lazy_preprocess True \
            --report_to wandb

    done
done



# 进行qa生成结果
for lr in 6e-5 ; do
    for epoch in 3; do  

    model_path="liuhaotian/llava-v1.5-7b"
    question_path="playground/data/test_questions.jsonl"
    image_folder="playground/data/images"
    N=8
    temperature=0.2

    base_answer_path="./output/llava100k_256_"${epoch}
    # Loop over each chunk/process
    for (( chunk_id=0; chunk_id<N; chunk_id++ ))
    do
        # Define the answer path for each chunk
        answer_path="${base_answer_path}_${temperature}_${chunk_id}.jsonl"
        if [ -f "$answer_path" ]; then
            rm "$answer_path"
        fi
        # Run the Python program in the background

        CUDA_VISIBLE_DEVICES="$chunk_id" python llava/eval/model_vqa_qa.py \
        --model-path "$model_path" --question-file "$question_path" --answers-file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --image-folder "$image_folder" --temperature "$temperature" \
        --pretrain_mm_mlp_adapter ./checkpoints/qa_tune100k_256_${epoch}/non_lora_trainables.bin \
        --lora_ckpt ./checkpoints/qa_tune100k_256_${epoch} \
        | tee qatest100k-256_${epoch}.txt &

    done

    # Wait for all background processes to finish
    wait

    merged_file="${base_answer_path}_${temperature}_merged.jsonl"
    if [ -f "$merged_file" ]; then
        rm "$merged_file"
    fi
    # Merge all the JSONL files into one
    #cat "${base_answer_path}"_*.jsonl > "${base_answer_path}_merged.jsonl"
    for ((i=0; i<N; i++)); do
    input_file="${base_answer_path}_${temperature}_${i}.jsonl"
    cat "$input_file" >> "${base_answer_path}_${temperature}_merged.jsonl"
    done
    # remove the unmerged files
    for (( chunk_id=0; chunk_id<N; chunk_id++ ))
    do
        # Define the answer path for each chunk
        answer_path="${base_answer_path}_${temperature}_${chunk_id}.jsonl"
        if [ -f "$answer_path" ]; then
            rm "$answer_path"
        fi
    done


    done
done

# 计算生成结果的正确率
python scripts/v1_5/qa_acc_cal.py \
    --ground_truth_file playground/data/test_answers.jsonl \
    --predictions_file ./output/llava100k_256_3_0.2_merged.jsonl



