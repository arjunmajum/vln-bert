import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer

from vilbert.vilbert import BertConfig

from utils.cli import get_parser
from utils.dataset.beam_dataset import BeamDataset
from utils.dataset.pano_features_reader import PanoFeaturesReader

from vln_bert import VLNBert

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=False)
    parser.add_argument(
        "--split",
        choices=["val_seen", "val_unseen", "test"],
        required=True,
        help="Dataset split for evaluation",
    )
    args = parser.parse_args()

    # force arguments
    args.num_beams = 1
    args.batch_size = 1

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=True)
    features_reader = PanoFeaturesReader(
        path="data/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        in_memory=args.in_memory,
    )
    dataset = BeamDataset(
        vln_path=f"data/task/R2R_{args.split}.json",
        beam_path=f"data/beamsearch/{args.split}.json",
        tokenizer=tokenizer,
        pano_features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_beams_strict=False,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=True,
    )
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BertConfig.from_json_file(args.config_file)
    model = VLNBert.from_pretrained(args.from_pretrained, config, default_gpu=True)
    model.cuda()
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- #
    # evaluation #
    # ---------- #

    with torch.no_grad():
        all_scores = eval_epoch(model, data_loader, args)

    # save scores
    scores_path = os.path.join(save_folder, f"scores_{args.split}.json")
    json.dump(all_scores, open(scores_path, "w"))
    logger.info(f"saving scores: {scores_path}")

    # covert scores into results format
    all_results = convert_scores(
        scores=all_scores,
        beam_path=f"data/beamsearch/{args.split}.json",
        add_exploration_path=args.split == "test",
    )

    # save results
    results_path = os.path.join(save_folder, f"results_{args.split}.json")
    json.dump(all_results, open(results_path, "w"))
    logger.info(f"saving results: {results_path}")


def eval_epoch(model, data_loader, args):
    device = next(model.parameters()).device

    model.eval()
    all_scores = []
    for batch in tqdm(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        batch_size = get_batch_size(batch)
        num_options = get_num_options(batch)
        instr_ids = get_instr_ids(batch)

        # get the model output
        output = model(*get_model_input(batch))
        vil_logit = output[0].view(batch_size, num_options)

        for instr_id, logit in zip(instr_ids, vil_logit):
            all_scores.append((instr_id, logit.tolist()))

    return all_scores


def convert_scores(all_scores, beam_path, add_exploration_path=False):
    beam_data = json.load(open(beam_path, "r"))
    instr_id_to_beams = {item["instr_id"]: item["ranked_paths"] for item in beam_data}
    if add_exploration_path:
        instr_id_to_exploration_path = {
            item["instr_id"]: item["exploration_path"] for item in beam_data
        }

    output = []
    for instr_id, scores in all_scores:
        idx = np.argmax(scores)
        beams = instr_id_to_beams[instr_id]
        trajectory = []
        if add_exploration_path:
            trajectory += instr_id_to_exploration_path[instr_id]
        trajectory += beams[idx]
        output.append({"instr_id": instr_id, "trajectory": trajectory})

    assert len(output) == len(beam_data)

    return output


# ------------- #
# batch parsing #
# ------------- #

# batch format:
# 0:target, 1:image_features, 2:image_locations, 3:image_mask, 4:image_targets,
# 5:image_targets_mask, 6:instr_tokens, 7:instr_mask, 8:instr_targets, 9:segment_ids,
# 10:co_attention_mask, 11:item_id


def get_model_input(batch):
    (
        _,
        image_features,
        image_locations,
        image_mask,
        _,
        _,
        instr_tokens,
        instr_mask,
        _,
        segment_ids,
        co_attention_mask,
        _,
    ) = batch

    # transform batch shape
    image_features = image_features.view(-1, image_features.size(2), 2048)
    image_locations = image_locations.view(-1, image_locations.size(2), 12)
    image_mask = image_mask.view(-1, image_mask.size(2))
    instr_tokens = instr_tokens.view(-1, instr_tokens.size(2))
    instr_mask = instr_mask.view(-1, instr_mask.size(2))
    segment_ids = segment_ids.view(-1, segment_ids.size(2))
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    return (
        instr_tokens,
        image_features,
        image_locations,
        segment_ids,
        instr_mask,
        image_mask,
        co_attention_mask,
    )


def get_batch_size(batch):
    return batch[1].size(0)


def get_num_options(batch):
    return batch[6].size(1)


def get_instr_ids(batch):
    instr_ids = batch[11]
    return [str(item[0].item()) + "_" + str(item[1].item()) for item in instr_ids]


if __name__ == "__main__":
    main()
