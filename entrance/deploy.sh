
deactivate
source ../instructblip-jailbreak/bin/activate
pip install -e ..
bash inference/instruct-blip/run_instruct-blip.sh |& tee ./logs/instruct-blip.log

deactivate
source ../minigpt4-jailbreak/bin/activate
pip install -e ..
bash inference/minigpt4/run_minigpt4-13b.sh |& tee ./logs/minigpt4-13b.log

deactivate
source ../qwen-vl-chat-jailbreak/bin/activate
pip install -e ..
bash inference/qwen-vl-chat/run_qwen-vl-chat.sh |& tee ./logs/qwen-vl-chat.log
#
deactivate
source ../cogvlm-jailbreak/bin/activate

bash inference/cogvlm/run_cogvlm.sh |& tee ./logs/cogvlm-our.log

deactivate
source ../fuyu-jailbreak/bin/activate

bash inference/fuyu/run_fuyu.sh |& tee ./logs/fuyu-our.log


deactivate
source ../llama-adapter-jailbreak/bin/activate

bash inference/llama-adapter/run_llama-adapter.sh |& tee ./logs/llama-adapter-our.log

deactivate
source ../llava-jailbreak/bin/activate

bash inference/llava/run_llavav1.5-7b.sh


deactivate
source ../mplug-owl2-jailbreak/bin/activate

bash inference/mplug-owl2/run_mplug-owl2.sh |& tee ./logs/mplug-owl2-our.log

deactivate
source ../instructblip-jailbreak/bin/activate

bash inference/instruct-blip/run_instruct-blip.sh |& tee ./logs/instruct-blip-our.log

deactivate
source ../minigpt4-jailbreak/bin/activate

bash inference/minigpt4/run_minigpt4-13b.sh |& tee ./logs/minigpt4-13b-our.log

deactivate
source ../qwen-vl-chat-jailbreak/bin/activate
bash inference/qwen-vl-chat/run_qwen-vl-chat.sh |& tee ./logs/qwen-vl-chat-our.log
