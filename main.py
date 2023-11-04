import click
import models
import importlib
from training import train
import torchvision
import torchvision.transforms as transforms
import json
import models
import os
import logging
from datetime import datetime
import torch


@click.command()
@click.option('--train_dir', type=str, default='data/raw/train', help='Path to traing dataset')
@click.option('--test_dir', type=str, default='data/raw/test', help='Path to test dataset')
@click.option('--seed', type=int, default=1, help='Random seed')
@click.option('--save_model', type=bool, default=False, help='Whether to save the model')
@click.option('--model_path', type=str, default='model.pt', help='Path to save/load the model')
@click.option('--load_model', type=bool, default=False, help='Whether to load the model')
@click.option('--evaluate', type=bool, default=True, help='Whether to evaluate the model')
@click.option('--config', type=str, default='config.json', help='Path to config file for specifying new model architectures')
@click.option('--log_dir', type=str, default='logs', help='Path to log directory')
@click.option('--loglevel', type=str, default='INFO', help='Log level')
def main(train_dir, test_dir, save_model, model_path, load_model, evaluate, config, log_dir, loglevel, seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)

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

    if load_model:
        model = models.load_model(model_path)
    else:
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
        print("Model: ",model)
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
                torch.save(model, model_path+"_"+now_str)
            else:
                torch.save(model, model_path)
    if evaluate:
        dataloader = torchvision.datasets.ImageFolder(
            test_dir, transform=input_transform)
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(
            len(dataloader.dataset.classes)))
        class_total = list(0. for i in range(len(dataloader.dataset.classes)))

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(
                    "cuda" if torch.cuda.is_available() else "cpu")
                labels = labels.to(
                    "cuda" if torch.cuda.is_available() else "cpu")
                outputs = model(images)
                _, predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                compare = predicted == labels
                correct += compare.sum().item()
                c = compare.squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        print('Accuracy of the network on the {} test images: {} %'.format(
            total, 100 * correct / total))
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(
                    "cuda" if torch.cuda.is_available() else "cpu")
                labels = labels.to(
                    "cuda" if torch.cuda.is_available() else "cpu")
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
        for i in range(len(dataloader.dataset.classes)):
            print('Accuracy of {} : {} %'.format(
                dataloader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
