echo "开始测试......"
python ./main.py \
    --data_config_path ./config/data_config/1.yaml \
    --model_config_path ./config/model_config/1.ymal \
    --train_config_path ./config/train_config/1.yaml \
    --model_name ASTGCN \
    --model_save_path ./model_states/ASTGCN_1.pkl \
    --result_save_dir ./result/ASTGCN_1 \


wait
python ./main.py \
    --data_config_path ./config/data_config/2.yaml \
    --model_config_path ./config/model_config/2.ymal \
    --train_config_path ./config/train_config/2.yaml \
    --model_name ASTGCN \
    --model_save_path ./model_states/ASTGCN_2.pkl \
    --result_save_dir ./result/ASTGCN_2 \

wait
python ./main.py \
    --data_config_path ./config/data_config/3.yaml \
    --model_config_path ./config/model_config/3.ymal \
    --train_config_path ./config/train_config/3.yaml \
    --model_name ASTGCN \
    --model_save_path ./model_states/ASTGCN_3.pkl \
    --result_save_dir ./result/ASTGCN_3 \

echo "结束测试......"
