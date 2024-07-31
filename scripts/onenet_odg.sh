echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/Weather_config.yaml" \
    --model_config_path "./config/model_config/ODGNet_model_config.yaml" \
    --train_config_path "./config/train_config/OneNet_ODG_train_config.yaml" \
    --model_name "onenet_odg" \
    --model_save_path "./model_states/OneNet_ODG_WTH/001.pkl" \
    --result_save_dir "./results/OneNet_ODG_WTH/001"
echo "结束测试......"