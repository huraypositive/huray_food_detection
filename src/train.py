import argparse
from ultralytics import YOLO
import wandb
import os

wandb.login(key=os.environ['INPUT_YOUR_WANDB_API_KEY']) # Enter the WANDB API KEY set in the OS

def train(ModelConfig):
    model = YOLO(ModelConfig['model_path'])
    results = model.train(data=ModelConfig['data_path'], project = ModelConfig['project'], name = ModelConfig['name'],
                        pretrained = ModelConfig['pretrained'], device = ModelConfig['device'],
                        epochs = ModelConfig['epoch'], patience = ModelConfig['patience'], warmup_epochs = ModelConfig['warmup_epochs'],
                        imgsz=ModelConfig['input_size'], batch = ModelConfig['batch'], single_cls = ModelConfig['single_cls'], 
                        lr0=ModelConfig['lr0'], lrf = ModelConfig['lrf'], optimizer = ModelConfig['optimizer'], cos_lr = ModelConfig['cos_lr'], workers=24)
    metrics = model.val()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--data_path', default = 'huray_food_detection/data/dataset_example.yaml', type = str, required = True)
    parser.add_argument('--project_name', type = str, required = True)
    parser.add_argument('--pretrained', type = bool, action = 'store_true')
    parser.add_argument('--device_ids', type = str, required = True)
    parser.add_argument('--epoch', type = int, required = True)
    parser.add_argument('--batch', type = int, required = True)
    parser.add_argument('--lr0', type = float, required = True)
    parser.add_argument('--optimizer', type = str, required = True)
    parser.add_argument('--cos_lr', type = bool, action = 'store_true')
    args = parser.parse_args()

    ModelConfig = {
            'patience': 100,
            'warmup_epochs': 5,
            'input_size': 640,
            'single_cls': True,
            'lrf': 0.01,
            }
    
    ModelConfig['model_path'] = args.model_path
    ModelConfig['data_path'] = args.data_path
    ModelConfig['project_name'] = args.project_name
    ModelConfig['name'] = f'{os.path.basename(args.model_path)}_{args.batch}_{args.lr0}_{args.optimizer}'
    ModelConfig['pretrained'] = args.pretrained
    ModelConfig['device'] = [id for id in args.device_ids.split(',')]
    ModelConfig['epoch'] = args.epoch
    ModelConfig['batch'] = args.batch
    ModelConfig['lr0'] = args.lr0
    ModelConfig['optimizer'] = args.optimizer
    ModelConfig['cos_lr'] = args.cos_lr
    
    train(ModelConfig)
    
