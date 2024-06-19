# AutoGeo
Code for AutoGeo.

# Environment
You can install the environment with this:
```
conda env create -f llava.yml
```

## Data Generation
To generate data, you can run:
```
cd data
python generate_datasets_multiprocess-reinforce.py
```

## Model Geometry Caption
After you generate data, you can tune model on the synthesized data by runing:
```
./scripts/v1_5/gc.sh
```
This script will tune the model and then evaluate the performance. 

## Model Geometry Questioning and Answering
After you tuned model on the synthesized data, you can further tune the model on QA data by runing:
```
./scripts/v1_5/gqa.sh
```

