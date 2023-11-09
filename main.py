import click
import pickle
import importlib
from training import train
import torchvision
import torchvision.transforms as transforms
import json
import os
import logging
import evaluate
from datetime import datetime
import torch
import torch.nn as nn


@click.command()
@click.option('--train_dir', type=str, default='data/raw/train', help='Path to traing dataset')
@click.option('--test_dir', type=str, default='data/raw/test', help='Path to test dataset')
@click.option('--seed', type=int, default=1, help='Random seed')
@click.option('--save_model', type=bool, default=False, help='Whether to save the model')
@click.option('--model_path', type=str, default='models/model.pth', help='Path to save/load the model')
@click.option('--load_model', type=bool, default=False, help='Whether to load the model')
@click.option('--eval', type=bool, default=True, help='Whether to evaluate the model')
@click.option('--config', type=str, default='config.json', help='Path to config file for specifying new model architectures')
@click.option('--log_dir', type=str, default='logs', help='Path to log directory')
@click.option('--loglevel', type=str, default='INFO', help='Log level')
def main(train_dir, test_dir, save_model, model_path, load_model, eval, config, log_dir, loglevel, seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    # Set up logging
    if loglevel not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError('Log level {} not supported'.format(loglevel))
    # Create logger and set its level to DEBUG
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create and configure a file handler to write all logs to a file
    log_file_name = 'log'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, log_file_name)):
        open(os.path.join(log_dir, log_file_name), 'a').close()
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    file_handler.setLevel(logging.DEBUG)  # Capture all logs
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create and configure a console handler to only display logs from WARNING level and above
    console_handler = logging.StreamHandler()
    # Only display WARNING and above
    console_handler.setLevel(getattr(logging, loglevel, None))
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    config = json.load(open(config))
    model_config = config['model']
    if "type" not in model_config.keys():
        raise ValueError("Model type not specified")
    found = False
    for file in os.listdir("models"):
        if file.endswith(".py"):
            module = importlib.import_module("models."+file[:-3])
            if hasattr(module, model_config['type']):
                model = getattr(module, model_config['type'])(
                    **model_config['args'])
                found = True
                break
    if hasattr(torchvision.models, model_config['type']):
        model = getattr(torchvision.models, model_config['type'])(
            **model_config['args'])
        found = True
    if not found:
        raise ValueError("Model {} not found".format(model_config['type']))
    if "device" in config:
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available")
        model = model.to(config["device"])
    else:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model: ", model)
    if load_model:
        model.load_state_dict(torch.load(model_path,map_location = torch.device(config["device"])))
    else:
        training_config = config['training']
        input_transform = []
        if "input_transform" in config:
            for transform in config["input_transform"]:
                if hasattr(transforms, transform["type"]):
                    input_transform.append(
                        getattr(transforms, transform["type"])(**transform["args"]))
                else:
                    raise ValueError(
                        "Transform {} not found".format(transform))
        input_transform = transforms.Compose(input_transform)
        output_transform = []
        if "output_transform" in config:
            for transform in config["output_transform"]:
                if hasattr(transforms, transform["type"]):
                    output_transform.append(
                        getattr(transforms, transform["type"])(**transform["args"]))
                else:
                    raise ValueError(
                        "Transform {} not found".format(transform))
        output_transform = transforms.Compose(output_transform)
        dataloader = torchvision.datasets.ImageFolder(
            train_dir, transform=input_transform, target_transform=output_transform)
        if "early_stopper" in training_config:
            if "train_data_split" not in config:
                raise ValueError(
                    "train_data_split must be specified for early stopping")
            training_config["early_stopper"] = train.EarlyStopper(
                **training_config["early_stopper"])
            tr_size, vl_size = config["train_data_split"].values()
            tr_size = int(tr_size*len(dataloader))
            vl_size = len(dataloader)-tr_size
            dataloader, valloader = torch.utils.data.random_split(
                dataloader, [tr_size, vl_size])
            training_config["valloader"] = torch.utils.data.DataLoader(
                valloader, batch_size=training_config['batch_size'],
                shuffle=training_config['shuffle'],
                num_workers=training_config['num_workers'])
        dataloader = torch.utils.data.DataLoader(
            dataloader, batch_size=training_config['batch_size'],
            shuffle=training_config['shuffle'],
            num_workers=training_config['num_workers'])
        training_config.pop('batch_size')
        training_config.pop('shuffle')
        training_config.pop('num_workers')
        training_config["device"] = config["device"]
        train.train(model, dataloader, **training_config)
        if save_model:
            if os.path.exists(model_path):
                # Get current date and time
                now = datetime.now()
                # Format it as a string
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                torch.save(model.state_dict(), model_path+"_"+now_str)
            else:
                torch.save(model.state_dict(), model_path)
                torch.save({
                'model_state_dict': model.state_dict()},
                model_path+"_state_dict")
    if eval:
        model.eval()
        dataloader = torchvision.datasets.ImageFolder(
            test_dir, transform=input_transform)
        dataloader = torch.utils.data.DataLoader(
            dataloader, batch_size=training_config['batch_size'],
            shuffle=training_config['shuffle'],
            num_workers=training_config['num_workers'])
        model.eval()
        val_res = evaluate.validate(model, dataloader, nn.CrossEntropyLoss(), config["device"])
        print("Validation loss: {}, Validation accuracy: {}".format(val_res.loss, val_res.accuracy))
        conf_matrix = evaluate.generate_conf_matrix(model, dataloader, config["device"],normalize=False)
        print(conf_matrix)



if __name__ == '__main__':
    main()
