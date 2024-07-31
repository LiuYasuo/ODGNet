echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/Weather_config.yaml" \
    --model_config_path "./config/model_config/ODGNet_model_config.yaml" \
    --train_config_path "./config/train_config/ODGNet_train_config.yaml" \
    --model_name "wo_trigger" \
    --model_save_path "./model_states/wo_trigger_WTH/001.pkl" \
    --result_save_dir "./results/wo_trigger_WTH/001"
echo "结束测试......"