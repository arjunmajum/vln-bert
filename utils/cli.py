import argparse


def get_parser() -> argparse.ArgumentParser:
    """ Return an argument parser with the standard VLN-BERT arguments.
    """
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument(
        "--in_memory",
        default=True,
        type=bool,
        help="Store the dataset in memory (default: True)",
    )
    parser.add_argument(
        "--bert_tokenizer",
        default="bert-base-uncased",
        type=str,
        help="Bert tokenizer model (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--config_file",
        default="data/config/bert_base_6_layer_6_connect.json",
        type=str,
        help="Model configuration file (default: data/config/bert_base_6_layer_6_connect.json)",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Load a pretrained model (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--max_instruction_length",
        default=60,
        type=int,
        help="The maximum number of instruction tokens used by the model (default: 60)",
    )
    parser.add_argument(
        "--max_path_length",
        default=8,
        type=int,
        help="The maximum number of viewpoints tokens used by the model (default: 8)",
    )
    parser.add_argument(
        "--max_num_boxes",
        default=101,
        type=int,
        help="The maximum number of regions used from each viewpoint (default: 101)",
    )
    parser.add_argument(
        "--num_beams",
        default=30,
        type=int,
        help="The fixed number of ranked paths to use in training (default: 30)"
    )
    parser.add_argument(
        "--output_dir",
        default="data/runs",
        type=str,
        help="The root output directory (default: data/runs)",
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="The name tag used for saving (default: '')",
    )
    parser.add_argument(
        "--training_mode",
        default='provided',
        choices=['sampled', 'provided', 'augmented'],
        help="The approach to collecting training paths (default: provided)",
    )
    parser.add_argument(
        "--masked_vision",
        action="store_true",
        help="Mask image regions during training (default: false)",
    )
    parser.add_argument(
        "--masked_language",
        action="store_true",
        help="Mask instruction tokens during training (default: false)",
    )
    parser.add_argument(
        "--no_ranking",
        action='store_true',
        help="Do not rank trajectories during training (default: false)",
    )
    parser.add_argument(
        "--num_epochs",
        default=20,
        type=int,
        help="Total number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="The size of one batch of training (default: 64)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=8,
        type=int,
        help="Number of step before a backward pass (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        default=4e-5,
        type=float,
        help="The initial learning rate (default: 4e-5)",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.04,
        type=float,
        help="Percentage of training to perform a linear lr warmup (default: 0.04)",
    )
    parser.add_argument(
        "--cooldown_factor",
        default=4.0,
        type=float,
        help="Multiplicative factor applied to the learning rate cooldown slope (default: 4.0)",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="The weight decay (default: 1e-2)"
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="Number of workers per gpu (default: 2)",
    )
    parser.add_argument(
        "--seed",
        default=101,
        type=int,
        help="Random number generator seed (default: 101)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Train on a small subset of the dataset (default: false)"
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local_rank for distributed training on gpus",
    )
    # fmt: on

    return parser
