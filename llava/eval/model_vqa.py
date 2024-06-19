import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # while(1):
    #     pass
    
    if args.lora_ckpt != "":
        model.load_adapter(args.lora_ckpt)
        print(f"adapter {args.lora_ckpt} loaded")
    
    if args.pretrain_vison != "":
        non_lora_state_dict = torch.load(args.pretrain_vison)
        new_state_dict = {}
        
        if args.pretrain_mm_mlp_adapter and args.lora_ckpt:
            prefix="base_model.model.model.vision_tower."
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().vision_tower.load_state_dict(new_state_dict) # , strict=False
        else:
            prefix="model.vision_tower."
            for key, value in non_lora_state_dict.items():
                # print(key)
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().vision_tower.load_state_dict(new_state_dict)
    
        # print(10*"==============\n")
        # model.get_model().mm_projector.load_state_dict(new_state_dict)
        
        
            
        print(f"vision {args.pretrain_vison} loaded")
        
    
    if args.pretrain_mm_mlp_adapter != "":
        
        non_lora_state_dict = torch.load(args.pretrain_mm_mlp_adapter)
        new_state_dict = {}
        
        if args.lora_ckpt:
            prefix="base_model.model.model.mm_projector."
            # prefix="1111111111111"
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
                # if "mm_projector" in key:
                #     print("!!!", key, new_key)
                print(key)
            model.get_model().mm_projector.load_state_dict(new_state_dict) # , strict=False
        else:
            # prefix="model.mm_projector."
            
            prefix="base_model.model.model.mm_projector."
            
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().mm_projector.load_state_dict(new_state_dict)
        
       
        
        # model.load_state_dict(new_state_dict)
        
        print(f"mmp {args.pretrain_mm_mlp_adapter} loaded")
    
    
    # # 获取LORA适配器的参数(打印可训练lora参数使用)
    # lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
    # print(lora_params)
    # # 计算LORA适配器的总参数量
    # total_lora_params = sum(param.numel() for param in lora_params)
    # print(f'Total LORA adapter parameters: {total_lora_params}')
    # exit()

    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt="\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\nQuestion:As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, DE parallel  BC, then the size of angle CED is ()\nChoices:\nA:40\u00b0\nB:60\u00b0\nC:120\u00b0\nD:140\u00b0"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--lora_ckpt", type=str, default='')
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default='')
    parser.add_argument("--pretrain_vison", type=str, default='')
    
    
    args = parser.parse_args()

    eval_model(args)