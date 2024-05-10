PORTIONS=8
NUM_GPUs=8

# LLaVAv1.5-7B
MODEL_NAME="LLaVAv1.5-7B"
MODEL_PATH=""
LM_PATH="None"
TASK_TYPE="vl"
SEED=42
#####################Iterate over testing dataset jsons#########################
BASE_DIR=""

#MODEL_LIST=("minigpt4-7b")
#METHOD_LIST=("clean-img" "FigStep" "VisualAdv-lp16-our" "ImageHijacks-lp16-our")
#MODEL_LIST=("instructblip-vicuna-7b")
#METHOD_LIST=("VisualAdv-lp32-our" "VisualAdv-unconstrained-our")

MODEL_LIST=("minigpt4-7b")
#METHOD_LIST=("ImageHijacks-lp16-our" "ImageHijacks-lp32-our" "ImageHijacks-unconstrained-our")
METHOD_LIST=("VisualAdv-lp32-our" "VisualAdv-unconstrained-our")
TOTAL_FILES=2
CURRENT_COUNT=0



for file in $(find $BASE_DIR -name "*.json"); do
  absolute_path=$(realpath $file)
  HARMFUL_BEHAVIORS_JSON_PATH=$absolute_path
  # Remove the file extension (.json)
  filename_without_extension="${file%.*}"

  SURROGATE_MODEL="None"
  for model in "${MODEL_LIST[@]}"
  do
    if [[ $filename_without_extension == *"$model"* ]]; then
      SURROGATE_MODEL=$model
      break
    fi
  done
  if [[ $SURROGATE_MODEL == "None" ]]; then
    continue
  fi
  ATTACK_METHOD="None"
  for method in "${METHOD_LIST[@]}"
  do
    if [[ $filename_without_extension == *"$method"* ]]; then
      ATTACK_METHOD=$method
      break
    fi
  done

  if [[ $ATTACK_METHOD == "None" ]]; then
    continue
  fi

  echo "PROGRESS  $CURRENT_COUNT / $TOTAL_FILES"
  echo "ATTACK_METHOD=$ATTACK_METHOD"
  echo "SURROGATE_MODEL=$SURROGATE_MODEL"
  echo "HARMFUL_BEHAVIORS_JSON_PATH=$HARMFUL_BEHAVIORS_JSON_PATH"
  python wrapper_for_evaluate.py \
    --portions $PORTIONS \
    --num_gpus $NUM_GPUs \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --harmful_behaviors_json_path $HARMFUL_BEHAVIORS_JSON_PATH \
    --task_type $TASK_TYPE \
    --attack_method $ATTACK_METHOD \
    --surrogate_model $SURROGATE_MODEL \
    --seed $SEED
  CURRENT_COUNT=$((CURRENT_COUNT+1))
done
#####################Iterate over testing dataset jsons#########################