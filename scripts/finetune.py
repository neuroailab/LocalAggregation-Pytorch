import os
import torch
from copy import deepcopy
from src.agents.agents import *
from src.utils.setup import process_config
from src.utils.utils import load_json


def run(config_path):
    config = process_config(config_path)
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if config.continue_exp_dir is not None:
        agent.logger.info("Found existing model... Continuing training!")
        agent.load_checkpoint('checkpoint.pth.tar',
                                checkpoint_dir=os.path.join(config.continue_exp_dir, 'checkpoints'),
                                load_model=True,
                                load_optim=True, load_epoch=True)

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

    run(args.config)
