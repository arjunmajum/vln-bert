import logging
import os
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from apex.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.tokenization import BertTokenizer

from vilbert.optimization import AdamW, WarmupLinearSchedule
from vilbert.vilbert import BertConfig
from vln_bert import VLNBert

from utils.cli import get_parser
from utils.dataset.beam_dataset import BeamDataset
from utils.dataset.pano_features_reader import PanoFeaturesReader
from utils.dataset.trajectory_dataset import TrajectoryDataset

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
    parser = get_parser(training=True)
    args = parser.parse_args()

    # validate command line arguments
    if not (args.masked_vision or args.masked_language) and args.no_ranking:
        parser.error(
            "No training objective selected, add --masked_vision, "
            "--masked_language, or remove --no_ranking"
        )

    # set seed
    if args.seed:
        seed = args.seed
        if args.local_rank != -1:
            seed += args.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # get device settings
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of synchronizing
        # nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        n_gpu = 1

    # check if this is the default gpu
    default_gpu = True
    if args.local_rank != -1 and dist.get_rank() != 0:
        default_gpu = False

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if default_gpu and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------ #
    # data loaders #
    # ------------ #

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=True)
    features_reader = PanoFeaturesReader(
        path="data/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        in_memory=args.in_memory,
    )

    if args.training_mode == "provided":
        if default_gpu:
            logger.info("using provided training trajectories")
        TrainDataset = BeamDataset
        vln_path = "data/task/R2R_train.json"
        beam_path = "data/beamsearch/beams_train.json"
    else:  # args.training_mode == "sampled":
        if default_gpu:
            logger.info("using sampled training trajectories")
        TrainDataset = TrajectoryDataset
        vln_path = "data/task/R2R_train.json"
        beam_path = None

    train_dataset = TrainDataset(
        vln_path=vln_path,
        beam_path=beam_path,
        tokenizer=tokenizer,
        pano_features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=4,
        num_beams_strict=False,
        training=True,
        masked_vision=args.masked_vision,
        masked_language=args.masked_language,
        default_gpu=default_gpu,
    )

    val_seen_dataset = BeamDataset(
        vln_path="data/task/R2R_val_seen.json",
        beam_path="data/beamsearch/beams_val_seen.json",
        tokenizer=tokenizer,
        pano_features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_beams_strict=True,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=default_gpu,
    )

    val_unseen_dataset = BeamDataset(
        vln_path="data/task/R2R_val_unseen.json",
        beam_path="data/beamsearch/beams_val_unseen.json",
        tokenizer=tokenizer,
        pano_features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_beams_strict=True,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=default_gpu,
    )

    # in debug mode only run on a subset of the datasets
    if args.debug:
        train_dataset = Subset(
            train_dataset,
            np.random.choice(range(len(train_dataset)), size=128, replace=False),
        )
        val_seen_dataset = Subset(
            val_seen_dataset,
            np.random.choice(range(len(val_seen_dataset)), size=64, replace=False),
        )
        val_unseen_dataset = Subset(
            val_unseen_dataset,
            np.random.choice(range(len(val_unseen_dataset)), size=64, replace=False),
        )

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        val_seen_sampler = SequentialSampler(val_seen_dataset)
        val_unseen_sampler = SequentialSampler(val_unseen_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        val_seen_sampler = DistributedSampler(val_seen_dataset)
        val_unseen_sampler = DistributedSampler(val_unseen_dataset)

    # adjust the batch size for distributed training
    batch_size = args.batch_size // args.gradient_accumulation_steps
    if args.local_rank != -1:
        batch_size = batch_size // dist.get_world_size()
    if default_gpu:
        logger.info(f"batch_size: {batch_size}")

    # create data loaders
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_seen_data_loader = DataLoader(
        val_seen_dataset,
        sampler=val_seen_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_unseen_data_loader = DataLoader(
        val_unseen_dataset,
        sampler=val_unseen_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BertConfig.from_json_file(args.config_file)
    if len(args.from_pretrained) == 0:  # hack for catching --from_pretrained ""
        model = VLNBert(config)
    else:
        model = VLNBert.from_pretrained(
            args.from_pretrained, config, default_gpu=default_gpu
        )

    if default_gpu:
        logger.info(
            f"number of parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # move/distribute model to device
    model.to(device)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)
        if default_gpu:
            logger.info("using distributed data parallel")
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        if default_gpu:
            logger.info("using data parallel")

    # ------------ #
    # optimization #
    # ------------ #

    # set parameter specific weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": args.weight_decay},
    ]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)

    # optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    # calculate learning rate schedule
    t_total = (
        len(train_data_loader) // args.gradient_accumulation_steps
    ) * args.num_epochs
    warmup_steps = args.warmup_proportion * t_total
    adjusted_t_total = warmup_steps + args.cooldown_factor * (t_total - warmup_steps)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=adjusted_t_total, last_epoch=-1,
    )

    # --------------- #
    # before training #
    # --------------- #

    # save the parameters
    if default_gpu:
        with open(os.path.join(save_folder, "config.txt"), "w") as fid:
            print(f"{datetime.now()}", file=fid)
            print("\n", file=fid)
            print(vars(args), file=fid)
            print("\n", file=fid)
            print(config, file=fid)

    # loggers
    if default_gpu:
        writer = SummaryWriter(
            logdir=os.path.join(save_folder, "logging"), flush_secs=30
        )
    else:
        writer = None

    # -------- #
    # training #
    # -------- #

    # run training
    if default_gpu:
        logger.info("starting training...")

    best_seen_success_rate, best_unseen_success_rate = 0, 0
    for epoch in range(args.num_epochs):
        # train for one epoch
        train_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            train_data_loader,
            writer,
            default_gpu,
            args,
        )

        # save the model every epoch
        if default_gpu:
            model_state = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
            model_path = os.path.join(save_folder, f"pytorch_model_{epoch + 1}.bin")
            torch.save(model_state, model_path)

        # run validation
        if not args.no_ranking:
            global_step = (epoch + 1) * len(train_data_loader)

            # run validation on the "val seen" split
            with torch.no_grad():
                seen_success_rate = val_epoch(
                    epoch,
                    model,
                    "val_seen",
                    val_seen_data_loader,
                    writer,
                    default_gpu,
                    args,
                    global_step,
                )
                if default_gpu:
                    logger.info(
                        f"[val_seen] epoch: {epoch + 1} success_rate: {seen_success_rate.item():.3f}"
                    )

            # save the model that performs the best on val seen
            if seen_success_rate > best_seen_success_rate:
                best_seen_success_rate = seen_success_rate
                if default_gpu:
                    best_seen_path = os.path.join(
                        save_folder, "pytorch_model_best_seen.bin"
                    )
                    shutil.copyfile(model_path, best_seen_path)

            # run validation on the "val unseen" split
            with torch.no_grad():
                unseen_success_rate = val_epoch(
                    epoch,
                    model,
                    "val_unseen",
                    val_unseen_data_loader,
                    writer,
                    default_gpu,
                    args,
                    global_step,
                )
                if default_gpu:
                    logger.info(
                        f"[val_unseen] epoch: {epoch + 1} success_rate: {unseen_success_rate.item():.3f}"
                    )

            # save the model that performs the best on val unseen
            if unseen_success_rate > best_unseen_success_rate:
                best_unseen_success_rate = unseen_success_rate
                if default_gpu:
                    best_unseen_path = os.path.join(
                        save_folder, "pytorch_model_best_unseen.bin"
                    )
                    shutil.copyfile(model_path, best_unseen_path)

    # -------------- #
    # after training #
    # -------------- #

    if default_gpu:
        writer.close()


def train_epoch(
    epoch, model, optimizer, scheduler, data_loader, writer, default_gpu, args
):
    device = next(model.parameters()).device

    model.train()
    for step, batch in enumerate(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        batch_size = get_batch_size(batch)
        num_options = get_num_options(batch)
        target = get_target(batch)
        linguistic_target = get_linguistic_target(batch)
        vision_target, vision_target_mask = get_vision_target(batch)

        # get the model output
        output = model(*get_model_input(batch))
        vil_logit = output[0].view(batch_size, num_options)
        vision_predictions = output[1].view(-1, output[1].shape[2])
        linguistic_predictions = output[2].view(-1, output[2].shape[-1])

        # calculate the masked vision loss
        vision_loss = torch.tensor(0, device=device)
        if args.masked_vision:
            vision_loss = F.kl_div(
                F.log_softmax(vision_predictions, dim=-1),
                vision_target,
                reduction="none",
            )
            vision_loss *= vision_target_mask.unsqueeze(-1).float()
            vision_loss = torch.sum(vision_loss) / max(1, torch.sum(vision_target_mask))

        # calculate the masked language loss
        linguistic_loss = torch.tensor(0, device=device)
        if args.masked_language:
            linguistic_loss = F.cross_entropy(
                linguistic_predictions, linguistic_target, ignore_index=-1
            )

        # calculate the trajectory re-ranking loss
        ranking_loss = torch.tensor(0, device=device)
        if not args.no_ranking:
            ranking_loss = F.cross_entropy(vil_logit, target, ignore_index=-1)

        # calculate the final loss
        loss = ranking_loss + vision_loss + linguistic_loss
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # backward pass
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # calculate accuracy
        correct = torch.sum(torch.argmax(vil_logit, 1) == target).float()

        # calculate accumulated stats
        reduced_vision_loss = vision_loss.detach()
        reduced_linguistic_loss = linguistic_loss.detach()
        reduced_ranking_loss = ranking_loss.detach()
        reduced_loss = loss.detach() * args.gradient_accumulation_steps
        reduced_correct = correct.detach()
        reduced_batch_size = torch.tensor(batch_size, device=device)

        # TODO: skip this `all_reduce` to speed-up runtime
        if args.local_rank != -1:
            reduced_vision_loss /= dist.get_world_size()
            reduced_linguistic_loss /= dist.get_world_size()
            reduced_ranking_loss /= dist.get_world_size()
            reduced_loss /= dist.get_world_size()
            dist.all_reduce(reduced_vision_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(reduced_linguistic_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(reduced_ranking_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(reduced_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(reduced_batch_size, op=dist.ReduceOp.SUM)

        # write stats to tensorboard
        if default_gpu:
            global_step = step + epoch * len(data_loader)
            writer.add_scalar("loss/train", reduced_loss, global_step=global_step)
            writer.add_scalar(
                "loss/vision", reduced_vision_loss, global_step=global_step
            )
            writer.add_scalar(
                "loss/language", reduced_linguistic_loss, global_step=global_step
            )
            writer.add_scalar(
                "loss/ranking", reduced_ranking_loss, global_step=global_step
            )
            writer.add_scalar(
                "accuracy/train",
                reduced_correct / reduced_batch_size,
                global_step=global_step,
            )
            writer.add_scalar(
                "learning_rate/train", scheduler.get_lr()[0], global_step=global_step
            )

        if default_gpu and args.debug:
            logger.info(
                f"[train] step: {step + 1} "
                f"vision loss: {reduced_vision_loss:0.2f} "
                f"language loss: {reduced_linguistic_loss:0.2f} "
                f"ranking loss: {reduced_ranking_loss:0.2f} "
                f"loss: {reduced_loss:0.2f} "
                f"accuracy: {reduced_correct / reduced_batch_size:0.2f} "
                f"lr: {scheduler.get_lr()[0]:0.1e}"
            )


def val_epoch(epoch, model, tag, data_loader, writer, default_gpu, args, global_step):
    device = next(model.parameters()).device

    # validation
    model.eval()
    stats = torch.zeros(3, device=device).float()
    for step, batch in enumerate(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        batch_size = get_batch_size(batch)
        num_options = get_num_options(batch)
        target = get_target(batch)

        # get the model output
        output = model(*get_model_input(batch))
        vil_logit = output[0].view(batch_size, num_options)

        # calculate loss
        loss = F.binary_cross_entropy_with_logits(vil_logit, target.float())

        # calculate success rate of the top scoring beam
        correct = torch.sum(
            target.gather(1, torch.argmax(vil_logit, 1).view(-1, 1))
        ).float()

        # accumulate
        stats[0] += loss
        stats[1] += correct
        stats[2] += batch_size

        if default_gpu and args.debug:
            logger.info(
                f"[{tag}] step: {step + 1} "
                f"running loss: {stats[0] / stats[2]:0.2f} "
                f"running success rate: {stats[1] / stats[2]:0.2f}"
            )

    if args.local_rank != -1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    # write stats to tensorboard
    if default_gpu:
        writer.add_scalar(
            f"loss/bce_{tag}", stats[0] / stats[2], global_step=global_step
        )
        writer.add_scalar(
            f"accuracy/sr_{tag}", stats[1] / stats[2], global_step=global_step
        )

    return stats[1] / stats[2]


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


def get_target(batch):
    return batch[0]


def get_linguistic_target(batch):
    return batch[8].view(-1)


def get_vision_target(batch):
    return (
        batch[4].view(-1, batch[4].shape[-1]),
        batch[5].view(-1),
    )


if __name__ == "__main__":
    main()
