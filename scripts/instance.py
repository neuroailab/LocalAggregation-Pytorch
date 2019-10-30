# from comet_ml import Experiment
# experiment = Experiment(api_key="l4wUtHmBfo1SqjrOHLsLtU0zN", project_name='general', workspace="honglin-chen")

from copy import deepcopy
from src.agents.agents import *
from src.utils.setup import process_config
from src.utils.utils import load_json
import os


def run(config_path, pre_checkpoint_dir):
    config = process_config(config_path)
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    if pre_checkpoint_dir is not None:
        # this will load both the weights and memory bank
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

    pre_checkpoint_dir = None
    if config_json['pretrained_exp_dir'] is not None:
        print("NOTE: found pretrained model...continue training")
        pre_checkpoint_dir = os.path.join(config_json['pretrained_exp_dir'], 'checkpoints')

    run(args.config, pre_checkpoint_dir)

