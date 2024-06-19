import argparse
import json
import re
import os

def extract_choice(q, answer, i):
    # print(answer)
    if isinstance(answer, list):
        answer=answer[0]
    pattern0 = r"Answer:([A-Da-d])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1).upper()
    
    pattern0 = r"Answer: ([A-Da-d])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1).upper()
    
    pattern0 = r"answer:([A-Da-d])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1).upper()
    
    pattern0 = r"answer: ([A-Da-d])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1).upper()
    
    pattern0 = r"answer choice ([A-Da-d])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1).upper()
    
    # Pattern 1: Therefore, the correct answer is option {choice}

    pattern1 = r"the correct answer is option ([A-Da-d])"
    match = re.search(pattern1, answer)
    if match:
        return match.group(1).upper()

    # Pattern 2: Therefore, option {choice} is selected
    pattern2 = r"option ([A-Da-d]) is selected"
    match = re.search(pattern2, answer)
    if match:
        return match.group(1).upper()

    # Pattern 3: Therefore, the answer is option {choice}
    pattern3 = r"answer is option ([A-Da-d])"
    match = re.search(pattern3, answer)
    if match:
        return match.group(1).upper()
    #
    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"answer is \(([A-Da-d])\)"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()

    pattern4 = r"answer is ([A-Da-d])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()

    pattern4 = r"Answer is ([A-Da-d])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()


    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"option ([A-Da-d])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()
    
    pattern4 = r"Option ([A-Da-d])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"([A-Da-d]) is the"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1).upper()

    if len(answer)==1:
        return answer.upper()

    # Pattern 5: find the solution from the last sentence
    sentences = answer.split(".")
    try:
        last_sentence = sentences[-2].strip()
    except:
        return None
    match = re.search(r'is ([A-Da-d])', last_sentence)
    if match:
        answer = match.group(1).upper()
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

def calculate_accuracy(predictions_file, ground_truth_file):
    with open(predictions_file, "r") as f_pred, open(ground_truth_file, "r") as f_gt:
        predictions = [json.loads(line) for line in f_pred]
        predictions_ids = [item["question_id"] for item in predictions]
        questions = [item["prompt"] for item in predictions]
        predictions = [item["text"] for item in predictions]
        ground_truth = [json.loads(line) for line in f_gt]
        ground_truth_dict = {}
        for grd in ground_truth:
            ground_truth_dict.update({grd["question_id"] : grd["text"]})
        ground_truth = [ground_truth_dict[i] for i in predictions_ids]
        # Extract choices from predictions
        # print(predictions)
        predicted_choices = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions))]
        none_num = len([x for x in predicted_choices if x is None])
        print(predicted_choices)
        print(f"nones : {none_num}")

        # Calculate accuracy
        total = len(ground_truth)
        correct = 0
        correct_list, wrong_list = [], []
        for q, gt, pred_c, pred in zip(questions, ground_truth, predicted_choices, predictions):
            # print(pred_c, gt)
            if pred_c == gt:
                # print("yes")
                correct+=1
                correct_list.append({"question": q, "pred": pred})
            else:
                wrong_list.append({"question": q, "pred": pred, "gt": gt})
        accuracy = correct / total * 100
        print(correct)
        print(total)
        if args.save_correct_wrong:
            correct_file, wrong_file = os.path.splitext(predictions_file)[0]+"_correct.json", os.path.splitext(predictions_file)[0]+"_wrong.json"
            with open(correct_file, "w") as f:
                json.dump(correct_list, f)
            with open(wrong_file, "w") as f:
                json.dump(wrong_list, f)
        if args.debug:
            with open(args.question_file, "r") as f_question:
                questions = [json.loads(line) for line in f_question]
            correct_result_indexs = [index for index,(pred, gt) in enumerate(zip(predicted_choices, ground_truth)) if pred == gt]
            correct_result = [(index, questions[index], predictions[index], ground_truth[index]) for index in correct_result_indexs]
            print(0)

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, default="/home/pirenjie/prj1/geo_llava/playground/data/unigeo/test_answers.jsonl")
    parser.add_argument("--predictions_file", type=str, default="/home/pirenjie/prj1/geo_llava/results/llava1.5_7b_geo_finetuned_test.jsonl")
    parser.add_argument("--question_file", type=str, default="/home/pirenjie/prj1/geo_llava/playground/data/unigeo/test_questions.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_correct_wrong", action="store_true")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.predictions_file, args.ground_truth_file)
    print(f"Accuracy: {accuracy:.2f}%")