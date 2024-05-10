
# Installation

```
cd red_teaming_gpt4
pip install -e .
```

# Code Structure
```
├── cache # dependencies of LLMs and MLLMs
├── datasets # dataset files 
│   ├── figstep-imgs.zip # imgs for FigStep
│   ├── json_for_lm # test json files for jailbreaking LLM
│   ├── json_for_lm_autodan
│   ├── json_for_lm_function
│   ├── json_for_lm_suffix
│   └── json_for_vlm_our # test json files for jailbreaking MLLMs
├── entrance # scripts to run jailbreak exps 
│   ├── calculate_asr.py # calculate ASR on MLLMs
│   ├── calculate_asr_lm.py # calculate ASR on LLMs
│   ├── deploy.sh # deploy jailbreak exps
│   ├── gpt4 # run GPT4 for inference
│   ├── inference # single inference file for MLLMs and LLMs
│   ├── run_eval_for_wrapper.py
│   └── wrapper_for_evaluate.py
└── src # model definitions, dataset loading, 
    └── jailbreak_benchmark
        ├── __init__.py
        ├── api.py
        ├── constants.py 
        ├── evaluation # evaluation 
        ├── methods # jailbreak methods 
        ├── models # MLLMs and LLMs
        ├── py.typed
        └── version.py

```


# Install Model Envs 
It is recommendated that each LLM and MLLM should have seperate running environment. For example, you can build the env for MiniGPT4 by 

```
cd red_teaming_gpt4 
python -m venv minigpt4-jailbreak --prompt="minigpt4-jailbreak"
source minigpt4-jailbreak/bin/activate 
# make sure the python and pip are directed to minigpt4-jailbreak
which python 
which pip 
pip install -r ./cache/vlms/minigpt4/requirements.txt # install requirements for minigpt4
pip install -e . # install jailbreak_benchmark 
```


# Jailbreak Experiments

```
cd red_teaming_gpt4/entrance
bash deploy.sh 
```

