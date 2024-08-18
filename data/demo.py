import sys
import os
sys.path.append('..')
from utils.loading_utils import load_definitions_and_rules
import graph as gh
import problem as pr
from clause_generation import * #CompoundClauseGen
import signal
from pretty_problem_statement_dict import * # 改为经过语料扩展的结果
import json
from data_augmentation.opencv import *
import shutil


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)
save_dir='./demo'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
else:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
if not os.path.isdir(save_dir+"/img"):
    os.mkdir(save_dir+"/img")
else:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
# Example entities and conditions for illustration purposes
defs_path = './defs.txt'
rules_path = './rules.txt'
complexity = 1
# Load definitions and rules
definitions, rules = load_definitions_and_rules(defs_path, rules_path)


data_desc=[]
cc_gen = CompoundClauseGen(definitions, complexity)

txt = cc_gen.generate_clauses()
# You can also manually call clauses by
# txt = "A B C = triangle A B C"

p = pr.Problem.from_txt(txt)

try:
    # Set an alarm for 2 seconds
    signal.alarm(2)
    # Code block to execute with timeout
    g, _ = gh.Graph.build_problem(p, definitions)
                    
    # Additionaly draw this generated problem
    gh.nm.draw_reinforce(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment],
        theme='',
        save_to=save_dir+f"/img/demo.jpg")

    data_desc.append(
        {
        "id": f"demo",
        "image": f"img/demo.jpg",
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
        
    signal.alarm(0)
except KeyboardInterrupt:
    sys.exit(0)
except:
    # print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Exiting ...")
    print('err occurred, retrying ...')
    sys.exit(0)

json_data = json.dumps(data_desc, indent=2)
with open(f"{save_dir}/data.json", "w") as file:
    file.write(json_data)
