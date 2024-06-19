import argparse
import json
import re
import os

def extract_choice(q, answer, i):
    if isinstance(answer, list):
        answer=answer[0]
    pattern0 = r"Answer:([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    
    pattern0 = r"Answer: ([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    
    pattern0 = r"answer:([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    
    pattern0 = r"answer: ([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    
    pattern0 = r"answer choice ([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    
    # Pattern 1: Therefore, the correct answer is option {choice}

    pattern1 = r"the correct answer is option ([A-D])"
    match = re.search(pattern1, answer)
    if match:
        return match.group(1)

    # Pattern 2: Therefore, option {choice} is selected
    pattern2 = r"option ([A-D]) is selected"
    match = re.search(pattern2, answer)
    if match:
        return match.group(1)

    # Pattern 3: Therefore, the answer is option {choice}
    pattern3 = r"answer is option ([A-D])"
    match = re.search(pattern3, answer)
    if match:
        return match.group(1)
    #
    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"answer is \(([A-D])\)"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)

    pattern4 = r"answer is ([A-D])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)

    pattern4 = r"Answer is ([A-D])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)


    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"option ([A-D])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)
    
    pattern4 = r"Option ([A-D])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)
    
    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"([A-D]) is the"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)

    if len(answer)==1:
        return answer.upper()

    # Pattern 5: find the solution from the last sentence
    sentences = answer.split(".")
    try:
        last_sentence = sentences[-2].strip()
    except:
        return None
    match = re.search(r'is ([A-D])', last_sentence)
    if match:
        answer = match.group(1)
        return answer
    # print(f"not found {i}")
    # print("#"*10)
    # print(f"question:\n{q}")
    # print("#"*10)
    # print(f"answer:\n{answer}")
    # print("#"*10)
    
        # print("called!!!")
        # return answer.upper()
    
    return None

def calculate_accuracy(predictions_file, ground_truth_file, predictions_file_1, predictions_file_2):
    with open(predictions_file, "r") as f_pred, open(predictions_file_1, "r") as f_pred1, open(predictions_file_2, "r") as f_pred2, open(ground_truth_file, "r") as f_gt:
        predictions = [json.loads(line) for line in f_pred]
        predictions_1 = [json.loads(line) for line in f_pred1]
        predictions_2 = [json.loads(line) for line in f_pred2]
        predictions_ids = [item["question_id"] for item in predictions]
        predictions_ids_1 = [item["question_id"] for item in predictions_1]
        predictions_ids_2 = [item["question_id"] for item in predictions_2]
        questions = [item["prompt"] for item in predictions]
        questions_1 = [item["prompt"] for item in predictions_1]
        questions_2 = [item["prompt"] for item in predictions_2]
        predictions = [item["text"] for item in predictions]
        predictions_1 = [item["text"] for item in predictions_1]
        predictions_2 = [item["text"] for item in predictions_2]
        ground_truth = [json.loads(line) for line in f_gt]
        ground_truth_dict = {}
        for grd in ground_truth:
            ground_truth_dict.update({grd["question_id"] : grd["text"]})
        ground_truth = [ground_truth_dict[i] for i in predictions_ids]
        # Extract choices from predictions
        predicted_choices = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions))]
        predicted_choices_1 = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions_1))]
        predicted_choices_2 = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions_2))]
        
        # none_num = len([x for x in predicted_choices if x is None])
        # # print(predicted_choices)
        # print(f"nones : {none_num}")

        # Calculate accuracy
        total = len(ground_truth)
        correct = 0
        correct_1 = 0
        correct_2 = 0
        correct_list, wrong_list = [], []
        correct_list_1, wrong_list_1 = [], []
        correct_list_2, wrong_list_2 = [], []
        for q, gt, pred_c, pred in zip(questions, ground_truth, predicted_choices, predictions):
            if pred_c == gt:
                correct+=1
                correct_list.append({"question": q, "pred": pred})
            else:
                wrong_list.append({"question": q, "pred": pred, "gt": gt})
        
        for q, gt, pred_c, pred in zip(questions, ground_truth, predicted_choices_1, predictions_1):
            if pred_c == gt:
                correct_1+=1
                correct_list_1.append({"question": q, "pred": pred})
            else:
                wrong_list_1.append({"question": q, "pred": pred, "gt": gt})
        
        for q, gt, pred_c, pred in zip(questions, ground_truth, predicted_choices_2, predictions_2):
            if pred_c == gt:
                correct_2+=1
                correct_list_2.append({"question": q, "pred": pred})
            else:
                wrong_list_2.append({"question": q, "pred": pred, "gt": gt})
        
        correct_list = [item['question'] for item in correct_list]
        wrong_list_1 = [item['question'] for item in wrong_list_1]
        wrong_list_2 = [item['question'] for item in wrong_list_2]
        
        common_questions = set(correct_list) & set(wrong_list_1) & set(wrong_list_2)
        
        print(common_questions)
        # result = []
        # for item in a:
        #     if item['question'] in common_questions:
        #         result.append(item)
        # for item in b:
        #     if item['question'] in common_questions:
        #         result.append(item)
        # for item in c:
        #     if item['question'] in common_questions:
        #         result.append(item)

        # print(correct_list, wrong_list_1, wrong_list_2)
        accuracy = correct / total * 100
        # print(correct)
        # print(total)
        # if args.save_correct_wrong:
        #     correct_file, wrong_file = os.path.splitext(predictions_file)[0]+"_correct.json", os.path.splitext(predictions_file)[0]+"_wrong.json"
        #     with open(correct_file, "w") as f:
        #         json.dump(correct_list, f)
        #     with open(wrong_file, "w") as f:
        #         json.dump(wrong_list, f)
        # if args.debug:
        #     with open(args.question_file, "r") as f_question:
        #         questions = [json.loads(line) for line in f_question]
        #     correct_result_indexs = [index for index,(pred, gt) in enumerate(zip(predicted_choices, ground_truth)) if pred == gt]
        #     correct_result = [(index, questions[index], predictions[index], ground_truth[index]) for index in correct_result_indexs]
        #     print(0)

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, default="/home/pirenjie/prj1/geo_llava/playground/data/unigeo/test_answers.jsonl")
    parser.add_argument("--predictions_file", type=str, default="/home/pirenjie/prj1/geo_llava/results/llava1.5_7b_geo_finetuned_test.jsonl")
    parser.add_argument("--predictions_file_1", type=str, default="/home/pirenjie/prj1/geo_llava/results/llava1.5_7b_geo_finetuned_test.jsonl")
    parser.add_argument("--predictions_file_2", type=str, default="/home/pirenjie/prj1/geo_llava/results/llava1.5_7b_geo_finetuned_test.jsonl")
    parser.add_argument("--question_file", type=str, default="/home/pirenjie/prj1/geo_llava/playground/data/unigeo/test_questions.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_correct_wrong", action="store_true")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.predictions_file, args.ground_truth_file, args.predictions_file_1, args.predictions_file_2)
    print(f"Accuracy: {accuracy:.2f}%")