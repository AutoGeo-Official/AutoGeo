import sys

sys.path.append('..')

from utils.loading_utils import load_definitions_and_rules

import random
import graph as gh
import problem as pr
from clause_generation import CompoundClauseGen
import signal

import multiprocessing
from pretty_problem_statement_dict import * # 改为经过语料扩展的结果

import json

from data_augmentation.opencv import *

class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)

need_proof = False
augment=False # 对图像进一步增强
test=False

# 描述数据
if not test:
    generate_num=100000 #50000 # 训练集
else:
    generate_num=1000 # 测试集
process_num=10
data_per_process = generate_num // process_num
remaining_data = generate_num % process_num
stage2=generate_num//5
stage3=3*generate_num//5   

if not test:
    if not need_proof:
        if generate_num==100000:
            save_dir = './dataset_reinforce_100k/'
        elif generate_num==50000:
            save_dir = './dataset_reinforce_50k/'
        elif generate_num==30000:
            save_dir = './dataset_reinforce_30k/'
        elif generate_num==10000:
            save_dir = './dataset_reinforce_10k/'
        else:
            save_dir = f'./dataset_reinforce_{generate_num}/'
    # else:
    #     save_dir = './dataset_reinforce_100k/'
else:
    if not need_proof:
        save_dir = './dataset_test/'
    # else:
    #     save_dir = './dataset_test/'
print(save_dir)

import os
folder_name = save_dir
import sys
# 创建文件夹
os.mkdir(folder_name)
os.mkdir(folder_name+"/img")


if not test: 
    random.seed(5)
else:
    random.seed(7)

# Example entities and conditions for illustration purposes
defs_path = './defs.txt'
rules_path = './rules.txt'
# Load definitions and rules
definitions, rules = load_definitions_and_rules(defs_path, rules_path)
lock = multiprocessing.Lock()

manager = multiprocessing.Manager()



def remove_uppercase_space(input_string):
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ=':
        input_string = input_string.replace(char, "")
    return input_string.replace(" ", "")

def gen_data(index_start, index_end, process_index, data_desc, num_analysis):
    
    i=index_start
    while i<index_end:
        print("Process", process_index, "processing ", i, "/", index_end)
        if i < stage2:
            cc_gen = CompoundClauseGen(definitions, 1)
        elif i < stage3:
            cc_gen = CompoundClauseGen(definitions, 2)
        else:
            cc_gen = CompoundClauseGen(definitions, 3)
        
        txt = cc_gen.generate_clauses()
        
        p = pr.Problem.from_txt(txt)
        
        try:
            # Set an alarm for 10 seconds
            signal.alarm(1)
            # Code block to execute with timeout
            g, _ = gh.Graph.build_problem(p, definitions)
                            
            # Additionaly draw this generated problem
            gh.nm.draw_reinforce(
                g.type2nodes[gh.Point],
                g.type2nodes[gh.Line],
                g.type2nodes[gh.Circle],
                g.type2nodes[gh.Segment],
                theme='',
                save_to=save_dir+f"/img/{i}.jpg")

            if augment:
                operations = [
                    # lambda x: noise(x),  # 噪声
                    # lambda x: gussian(x),  # 模糊
                    # lambda x: light(x),  # 光线
                    lambda x: occlude(x, occluder_ratio=0.25),
                    lambda x: x,  # 什么都不做
                ]
                # 随机选择并执行一个操作
                selected_op = random.choice(operations)
                img = cv2.imread(save_dir+f"/img/{i}.jpg")
                #img = cv2.resize(img,(224,224))
                img = selected_op(img)
                cv2.imwrite(save_dir+f"/img/{i}.jpg", img)

            with lock:
                data_desc.append(
                    {
                    "id": f"{i}",
                    "image": f"img/{i}.jpg",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Render a clear and concise description of a image about geometric shapes.\n<image>"
                        },
                        {
                            "from": "gpt",
                            "value": gen_nl(txt)
                        }
                    ],
                    "clause": [remove_uppercase_space(clause_item) for clause_item in txt.split(";")]
                }
                )
                
                for clause_item in [remove_uppercase_space(clause_item) for clause_item in txt.split(";")]:
                    if clause_item in num_analysis:
                        num_analysis[clause_item] += 1
                    else:
                        num_analysis[clause_item] = 1
                
            signal.alarm(0)
            i+=1
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            # print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Exiting ...")
            print('err occurred, retrying ...')
            continue


def main():
    pool = multiprocessing.Pool(process_num)  # 创建进程池
    data_desc = manager.list()
    num_analysis= manager.dict()
    
    for process_index in range(process_num):
        gen_start=process_index*data_per_process
        gen_end=process_index*data_per_process+(data_per_process if process_index!=process_num-1 else data_per_process+remaining_data)
        # print(gen_start, gen_end)
        pool.apply_async(gen_data, (gen_start, gen_end, process_index, data_desc, num_analysis, ))

    pool.close()
    pool.join()
    # print(data_desc)
    data_desc=list(data_desc)
    json_data = json.dumps(data_desc, indent=2)
    # print(json_data)
    with open(f"{save_dir}/data.json", "w") as file:
        file.write(json_data)

    print("process done.")
    sorted_data = sorted(num_analysis.items(), key=lambda x: x[1], reverse=True)
    print(sorted_data)
    with open("sorted_data.json", "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    print(len(sorted_data))
if __name__ == "__main__":
    main()
