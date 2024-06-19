import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.eval.caption_score import *
from llava.eval.caption_score import score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)


    if "mathverse" in annotation_file.lower() or "math_verse" in annotation_file.lower():
        # 处理修改过的mathverse数据集
        annotations = [json.loads(line) for line in open(annotation_file, 'r')]
        annotations = {(annotation['problem_index']): annotation['desc'] for annotation in annotations}
    else:
        # 处理自己的数据集
        annotations = json.load(open(annotation_file))
        annotations = {(annotation['id']): annotation['conversations'][1]['value'] for annotation in annotations}
        # print(annotations)
    
    results = [json.loads(line) for line in open(result_file)]
    # print(result_file)
    print(results)
    
    gt_list = {}
    pred_list = {}
    for result in results:
        if "mathverse" or "math_verse" or "desc" in annotation_file.lower():
            annotation = annotations[str(result['question_id'])]
        else:
            annotation = annotations[str(result['question_id'])] # prompt_processor(result['prompt'])
        
        gt_list[str(result['question_id'])]=[annotation]
        pred_list[str(result['question_id'])]=[result['text']] #  #  if result['text']!="" else "null"
        # pred_list.append({
        #     "pred_answer": result['text'],
        #     "gt_answers": annotation,
        # })
    print(gt_list)
    print("========")
    print(pred_list)
    final_scores=score(gt_list, pred_list)
    print(final_scores)
    # evaluator = TextVQAAccuracyEvaluator()
    # print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
