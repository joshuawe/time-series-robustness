from ast import arg
from data.human_activity_recognition_data_pipeline import load_postprocessed_dataset
from models.lstm_model import Model as LSTMModel
import torch
from torch.utils.data import DataLoader, random_split
import os
import yaml
import argparse

# set current working directory
# os.chdir("./time-series-robustness")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file, model_name, version):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)['experiments']
    for item in config:
        if item.get("version") == version:
            return item[model_name]
    raise ValueError(f"Version {version} not found in config file {config_file}")

def LSTMModel_experiment(config_file = "./src/config.yaml", version='1.0'):

    # load experiments parameters
    config = load_config(config_file, "lstm_model", version)

    train_set, test_set = load_postprocessed_dataset(config)

    datasets = {}

    # split train set into train and validation set
    train_set, val_set = random_split(
                    train_set, [int(0.8*len(train_set)), 
                    len(train_set)-int(0.8*len(train_set))]
                    )
    
    batch_size = config['training']['batch_size']
    datasets["train"] = DataLoader(train_set, batch_size=batch_size)
    datasets["val"] = DataLoader(val_set, batch_size=batch_size)
    datasets["test"] = DataLoader(test_set, batch_size=batch_size)
    
    config['datasets'] = datasets
    model = LSTMModel(config).to(device)

    # train model
    model.train() # set model to train mode
    model.training_loop()

    # evaluate model
    model.eval() # set model to evaluation mode
    model.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./src/config.yaml")
    parser.add_argument('--exp_version', type=str, default='1.0')
    args = parser.parse_args()
    print("Experiment version: ", args.exp_version)
    print("Config file: ", args.config_file)
    
    LSTMModel_experiment(args.config_file, args.exp_version)