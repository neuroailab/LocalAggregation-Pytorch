import os
import torch
from copy import deepcopy
from src.agents.agents import *
from src.utils.setup import process_config, _process_config
from src.utils.utils import load_json


def run(config_path, ir_checkpoint_dir=None, pre_checkpoint_dir=None):
    config = process_config(config_path)
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if ir_checkpoint_dir is not None:
        agent.load_checkpoint('checkpoint.pth.tar', ir_checkpoint_dir, load_memory_bank=True,
                              load_model=True, load_optim=False, load_epoch=False)

    if pre_checkpoint_dir is not None:
        agent.load_checkpoint('checkpoint.pth.tar', pre_checkpoint_dir, load_memory_bank=True,
                              load_model=True, load_optim=True, load_epoch=True)

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    args = parser.parse_args()

    config_json = load_json(args.config)

    ir_checkpoint_dir = None
    pre_checkpoint_dir = None

    if config_json['pretrained_exp_dir'] is not None:
        print("NOTE: found pretrained model...Continue training")
        pre_checkpoint_dir = os.path.join(config_json['pretrained_exp_dir'], 'checkpoints')

    # If pre_checkpoint_dir already exits, ignore ir_checkpoint_dir
    if pre_checkpoint_dir is None and config_json['instance_exp_dir'] is not None:
        print("NOTE: found IR model...")
        ir_checkpoint_dir = os.path.join(config_json['instance_exp_dir'], 'checkpoints')

    run(args.config, ir_checkpoint_dir=ir_checkpoint_dir, pre_checkpoint_dir=pre_checkpoint_dir)
