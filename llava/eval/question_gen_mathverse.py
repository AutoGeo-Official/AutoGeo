import argparse
import json


def data_gen(output_file_path, refer_file_path):
    
    with open(refer_file_path, 'r') as f:
        ref_data = [json.loads(line.strip()) for line in f]
            # 现在 data 就是一个 Python 字典对象
    
    # question_num=len(ref_data)
    
    data = []
    for item in ref_data:
        data.append(
            {
            "question_id": item.get("problem_index"),
            "image": item.get("image"),
            "text": "Render a clear and concise description of a image about geometric shapes.\n",
            }
        )

    with open(output_file_path, "w") as file:
        file.write("")
    # 打开文件，以追加模式写入数据
    with open(output_file_path, "a") as file:
        # 遍历数据列表
        for item in data:
            # 将字典对象转换为JSON字符串
            json_str = json.dumps(item)
            # 写入JSON字符串，并添加换行符
            file.write(json_str + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, default="data/questions_mathverse/question.jsonl")
    
    parser.add_argument("--refer_file_path", type=str, default="data/mathverse/testdata-planegeo-desc.jsonl")
    
    args = parser.parse_args()
    # 定义要写入的文件路径
    output_file_path = args.output_file_path
    refer_file_path=args.refer_file_path

    data_gen(output_file_path, refer_file_path)


