import os
import random
import argparse
import configparser

import numpy as np
from torch.backends import cudnn
import torch

from solver import Solver
import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config['model_save_path']):
        os.makedirs(config['model_save_path'])
    solver = Solver(config)

    if config['mode'] == 'train':
        solver.train()
    elif config['mode'] == 'test':
        solver.test()
    else:
        raise ValueError(f"Unrecognized mode: {config['mode']}")

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    fileconfig = configparser.ConfigParser()
    fileconfig.read(args.config)

    config = {
        'lr': float(fileconfig['train']['lr']),
        'gpu': str(fileconfig['train']['gpu']),
        'num_epochs': int(fileconfig['train']['epoch']),
        'anormly_ratio': float(fileconfig['train']['ar']),
        'batch_size': int(fileconfig['train']['bs']),
        'seed': int(fileconfig['train']['seed']),
        'win_size': int(fileconfig['data']['ws']),
        'input_c': int(fileconfig['data']['ic']),
        'output_c': int(fileconfig['data']['oc']),
        'dataset': str(fileconfig['data']['ds']),
        'data_path': str(fileconfig['data']['dp']), 
        'd_model': int(fileconfig['param']['d']),
        'e_layers': int(fileconfig['param']['l']),
        'fr': float(fileconfig['param']['fr']),
        'tr': float(fileconfig['param']['tr']),
        'seq_size': int(fileconfig['param']['ss']),
        'mode': str(fileconfig['model']['mode']),
        'model_save_path': str(fileconfig['model']['msp'])
    }

    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True

    sys.stdout = Logger("result/" + config['dataset'] + ".log", sys.stdout)
    print('------------ Options -------------')
    for k, v in sorted(config.items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')
    
    main(config)



