import os
import argparse
import torch

from src.config import Config
from src.utils import set_seed
from src.pet_synthesis import test_generator
from src.pet_classification import test_classifier


if __name__ == "__main__":

    # Configuration
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('--config', type=str, default='syn_config',
                        choices=['syn_config', 'cls_config'],
                        help='configuration filename')
    args = parser.parse_args()
    config = Config(filename=args.config, mode='test')

    # Device
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    # Random seed
    set_seed(config.MANUAL_SEED)

    # Main
    if args.config == 'syn_config':
        test_generator(config)
    elif args.config == 'cls_config':
        test_classifier(config)
    else:
        raise ValueError('Invalid config file')
