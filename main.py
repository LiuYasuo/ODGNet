import json
import os
import sys
import random
import yaml
import argparse
import numpy as np
import torch

import trainers
import models
import datasets
import data_preprocessors

from utils import scaler
from torch.utils.data import DataLoader



def load_config(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config



def main(args):
    result_save_dir_path = args.result_save_dir_path
    model_save_path = args.model_save_path
    if not os.path.exists(result_save_dir_path):
        os.makedirs(result_save_dir_path)
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    train_config = load_config(args.train_config_path)
    model_config = load_config(args.model_config_path)
    data_config = load_config(args.data_config_path)

    # basic config
    random_seed = train_config["random_seed"]
    device = torch.device(train_config["device"])
    batch_size = train_config["batch_size"]
    if train_config.get("test_bsz", None) != None:
        online_batch_size = train_config["test_bsz"]
    else:
        online_batch_size = batch_size
    load_checkpoint = train_config["load_checkpoint"]

    # set random seeds for reproducibility

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    # ----------------------- Load data ------------------------
    data_preprocessor_class = getattr(
        sys.modules["data_preprocessors"], data_config["data_preprocessor_name"]
    )
    data_preprocessor = data_preprocessor_class(**data_config["data_preprocessor_params"])
    preprocessed_data = data_preprocessor.preprocess()

    train_data, valid_data, test_data = data_preprocessor.split_data(preprocessed_data)

    # update model & trainer params
    updated_dataset_params = {**data_config["dataset_params"], **data_preprocessor.update_dataset_params}
    updated_model_params = {**model_config[args.model_name], **data_preprocessor.update_model_params}
    updated_trainer_params = {**train_config["trainer_params"], **data_preprocessor.update_trainer_params}

    # scale data
    scaler_class = getattr(sys.modules["utils.scaler"], data_config["scaler_name"])
    scaler = scaler_class(**data_config["scaler_params"])
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)


    # dataset
    dataset_class = getattr(sys.modules["datasets"], data_config["dataset_name"])
    train_dataset = dataset_class(train_data, **updated_dataset_params)
    valid_dataset = dataset_class(valid_data, **updated_dataset_params)
    test_dataset = dataset_class(test_data, **updated_dataset_params)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, **data_config["dataloader_params"]["train"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, **data_config["dataloader_params"]["valid"]
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=online_batch_size, **data_config["dataloader_params"]["test"]
    )

    # ------------------------- Model ---------------------------

    # model
    model_class = getattr(sys.modules["models"], args.model_name)
    model = model_class(**updated_model_params)
    model.to(device)

    # ------------------------- Trainer -------------------------

    # Optimizer
    optimizer_class = getattr(
        sys.modules["torch.optim"], train_config["optimizer_name"]
    )
    optimizer = optimizer_class(model.parameters(), **train_config["optimizer_params"])

    # scheduler
    scheduler_class = getattr(
        sys.modules["torch.optim.lr_scheduler"], train_config["scheduler_name"]
    )
    scheduler = scheduler_class(optimizer, **train_config["scheduler_params"])

    # trainer
    trainer_class = getattr(sys.modules["trainers"], train_config["trainer_name"])
    trainer = trainer_class(
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        **updated_trainer_params
    )

    # load checkpoint
    if load_checkpoint:
        trainer.load_checkpoint()

    # ------------------------- Train & Test ------------------------
    config = {
        "args": vars(args),
        "train_config": train_config,
        "model_config": model_config[args.model_name],
        "data_config.yaml": data_config,
    }
    print("Configuration: ", config)
    with open(os.path.join(result_save_dir_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("Start training.")

    epoch_results = trainer.train(train_dataloader, valid_dataloader)
    test_result, y_pred, y_true = trainer.test(test_dataloader)

    # save y_pred, y_true to self.result_save_dir/y_pred.npy, y_true.npy
    np.save(os.path.join(result_save_dir_path, "test_y_pred.npy"), y_pred)
    np.save(os.path.join(result_save_dir_path, "test_y_true.npy"), y_true)

    # save results
    result = {
        "config": config,
        "test_result": test_result,
        "epoch_results": epoch_results,
    }
    with open(os.path.join(result_save_dir_path, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="./config/train_config/ODGNet_train_config.yaml",
        help="Config path of Trainer",
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/model_config/ODGNet_model_config.yaml",
        help="Config path of models",
    )

    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/data_config/Electricity_config.yaml",
        help="Config path of Data",
    )
    parser.add_argument("--model_name", type=str, default="odgnet", help="Model name")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./model_states/ODGNet/001.pkl",
        help="Model save path",
    )

    parser.add_argument(
        "--result_save_dir_path",
        type=str,
        default="./results/ODGNet/001",
        help="Result save path",
    )
    args = parser.parse_args()
    main(args)


