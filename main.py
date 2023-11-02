import click
import models
import training.train as train
import torchvision, torchvision.transforms as transforms
import json
import os
from datetime import datetime
import torch


@click.command()
@click.option('--train_dir', type=str, default='data/train', help='Path to traing dataset')
@click.option('--test_dir', type=str, default='data/test', help='Path to test dataset')
@click.option('--seed', type=int, default=1, help='Random seed')
@click.option('--save_model', type=bool, default=False, help='Whether to save the model')
@click.option('--model_path', type=str, default='model.pt', help='Path to save/load the model')
@click.option('--model', type=str, default='resnet18', help='Model to use')
@click.option('--load_model', type=bool, default=False, help='Whether to load the model')
@click.option('--plot_loss', type=bool, default=True, help='Whether to plot the training loss')
@click.option('--evaluate', type=bool, default=True, help='Whether to evaluate the model')
@click.option('--config', type=str, default='config.json', help='Path to config file for specifying new model architectures')
@click.option('--log_dir', type=str, default='logs', help='Path to log directory')
def main(train_dir, test_dir, save_model, model_path, model, load_model, evaluate, config, log_dir):
    if load_model:
        model = models.load_model(model_path)
    else:
        if not hasattr(models,model):
            raise ValueError('Model {} not found'.format(model))
        config = json.load(open(config))
        model_config = config['model']
        model = getattr(models,model)(**model_config)
        if "device" in config:
            if config["device"] == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA not available")
            model = model.to(config["device"])
        else:
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        training_config = config['training']
        input_transform = []
        if "input_transform" in config:
            for transform in config["input_transform"]:
                if hasattr(transforms,transform):
                    input_transform.append(getattr(transforms,transform))
                else:
                    raise ValueError("Transform {} not found".format(transform))
        input_transform = transforms.Compose(input_transform)
        output_transform = []
        if "output_transform" in config:
            for transform["type"] in config["output_transform"]:
                if hasattr(transforms,transform):
                    output_transform.append(getattr(transforms,transform)(**transform["args"]))
                else:
                    raise ValueError("Transform {} not found".format(transform))
        output_transform = transforms.Compose(output_transform)
  
        dataloader = torchvision.datasets.ImageFolder(train_dir, transform=input_transform, target_transform=output_transform)
        train(model, dataloader, **training_config)
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
        dataloader = torchvision.datasets.ImageFolder(test_dir, transform=input_transform)
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(dataloader.dataset.classes)))
        class_total = list(0. for i in range(len(dataloader.dataset.classes)))

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
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
        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
        for i in range(len(dataloader.dataset.classes)):
            print('Accuracy of {} : {} %'.format(dataloader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()
        