import argparse

from ray.air.config import RunConfig, ScalingConfig
from ray.train.examples.pytorch.torch_fashion_mnist_example import train_func
from ray.train.torch import TorchTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

import pickle
import mlflow

import torch.optim as optim
import time
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module): # extend nn.Module class of nn
    def __init__(self):
        super().__init__() # super class constructor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5))
        self.batchN1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5,5))
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.batchN2 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
        
        
    def forward(self, t): # implements the forward method (flow of tensors)
        
        # hidden conv layer 
        t = self.conv1(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)
        t = self.batchN1(t)
        
        # hidden conv layer
        t = self.conv2(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)
        
        # flatten
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.batchN2(t)
        t = self.fc2(t)
        t = F.relu(t)
        
        # output
        t = self.out(t)
        
        return t

def main(num_workers=1, use_gpu=False):
    train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4}
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(
            callbacks=[MLflowLoggerCallback(experiment_name="train_fashion_mnist",
            save_artifact=True)]
        ),
    )
    results = trainer.fit()
    print("Final Results: ", results)
    print("Final Result Path: ", results.path)
    print("Final metrics: ", results.metrics)

    checkpoint_path = results.path
    print("Checkpoint Path: ", checkpoint_path)
    checkpoint = pickle.load(open(f"{checkpoint_path}/params.pkl", "rb"))
    cnn_model = Network() # init model
    print(cnn_model) # print model structure

    setup_mlflow(train_loop_config, create_experiment_if_not_exists=True, experiment_name="mnist", artifact_location = f"{results.path}")
    mlflow.pytorch.save_model(cnn_model, f"{results.path}/model" )
    mlflow.pytorch.log_model(cnn_model,f"model" )
    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )
    args, _ = parser.parse_known_args()

    import ray

    if args.smoke_test:
        ray.init(num_cpus=1)
        args.num_workers = 1
        args.use_gpu = False
    else:
        ray.init(address=args.address)
    main(num_workers=args.num_workers, use_gpu=args.use_gpu)
