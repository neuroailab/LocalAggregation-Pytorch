import os
import sys
import torch
import logging
import numpy as np
from pprint import pprint
from dotmap import DotMap
from logging import Formatter
from logging.handlers import RotatingFileHandler
from time import strftime, localtime, time

from src.utils.utils import load_json, save_json


def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def process_config(config_path, override_dotmap=None):
    config_json = load_json(config_path)
    return _process_config(config_json, override_dotmap=override_dotmap)


def _process_config(config_json, override_dotmap=None):
    """
    Processes config file:
        1) Converts it to a DotMap
        2) Creates experiments path and required subdirs
        3) Set up logging
    """
    config = DotMap(config_json)
    if override_dotmap is not None:
        config.update(override_dotmap)

    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    # if config.pretrained_exp_dir is not None:
    #     # don't make new dir more continuing training
    #     exp_dir = config.pretrained_exp_dir
    #     print("[INFO]: Continuing from previously finished training at %s." % exp_dir)
    # else:
    exp_base = config.exp_base

    if config.debug:
        exp_dir = os.path.join(exp_base, "experiments",
                               config.exp_name, 'debug')
    else:
        if config.pretrained_exp_dir is not None and isinstance(config.pretrained_exp_dir, str):
            # don't make new dir more continuing training
            exp_dir = config.pretrained_exp_dir
            print('[INFO]: Backup previously trained model and config json')
            os.system("cp %s/config.json %s/prev_config.json" % (exp_dir, exp_dir))
            os.system("cp %s/checkpoints/checkpoint.pth.tar %s/checkpoints/prev_checkpoint.pth.tar" % (exp_dir, exp_dir))
            os.system("cp %s/checkpoints/model_best.pth.tar %s/checkpoints/prev_model_best.pth.tar" % (exp_dir, exp_dir))
        elif config.continue_exp_dir is not None and isinstance(config.continue_exp_dir, str):
            exp_dir = config.continue_exp_dir
            print('[INFO]: Backup previously trained model and config json')
            os.system("cp %s/config.json %s/prev_config.json" % (exp_dir, exp_dir))
            os.system(
                "cp %s/checkpoints/checkpoint.pth.tar %s/checkpoints/prev_checkpoint.pth.tar" % (exp_dir, exp_dir))
            os.system(
                "cp %s/checkpoints/model_best.pth.tar %s/checkpoints/prev_model_best.pth.tar" % (exp_dir, exp_dir))
        else:
            if config.exp_id is None:
                config.exp_id = strftime('%Y-%m-%d--%H_%M_%S', localtime())
            exp_dir = os.path.join(exp_base, "experiments",
                                   config.exp_name, config.exp_id)

    # create some important directories to be used for the experiment.
    config.summary_dir = os.path.join(exp_dir, "summaries/")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")

    makedirs([config.summary_dir, config.checkpoint_dir,
              config.out_dir, config.log_dir])

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Experiment directory is located at %s" % exp_dir)

    logging.getLogger().info(
        "Configurations and directories successfully set up.")
    return config


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        '{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        '{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    try:
        call(["nvcc", "--version"])
    except:
        pass
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))
