import argparse
import json
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        "-s",
        "--scores",
        type=str,
        help="path to VLN-BERT scores",
    )
    parser.add_argument(
        "-b",
        "--beam-scores",
        type=str,
        help="path to beamsearch scores",
    )
    # fmt: on
    args = parser.parse_args()

    if "val_seen" in args.scores:
        args.split = "val_seen"
    elif "val_unseen" in args.scores:
        args.split = "val_unseen"
    elif "test" in args.scores:
        args.split = "test"
    else:
        raise ValueError("Could not infer dataset split")

    return args


def load_vln_data(split):
    path = f"data/task/R2R_{split}.json"
    data = json.load(open(path, "r"))
    instr_id_to_goal, instr_id_to_scan = {}, {}
    for item in data:
        for idx, _ in enumerate(item["instructions"]):
            instr_id = f"{item['path_id']}_{idx}"
            instr_id_to_scan[instr_id] = item["scan"]
            instr_id_to_goal[instr_id] = item["path"][-1]
    return instr_id_to_goal, instr_id_to_scan


def load_distances(scans):
    distances = {}
    for scan in scans:
        with open(f"data/distances/{scan}_distances.json", "r") as fid:
            distances[scan] = json.load(fid)
    return distances


def load_results_data(path):
    data = json.load(open(path, "r"))
    instr_id_to_scores = {item[0]: item[1] for item in data}
    return instr_id_to_scores


def load_beamsearch_data(path):
    data = json.load(open(path, "r"))
    instr_id_to_beams = {item["instr_id"]: item["ranked_paths"] for item in data}
    instr_id_to_exploration_path = {
        item["instr_id"]: item["exploration_path"] for item in data
    }
    return instr_id_to_beams, instr_id_to_exploration_path


def get_speaker_score(beam):
    speaker_score = sum(map(float, beam["speaker_scores"])) / len(
        beam["speaker_scores"]
    )
    return speaker_score


def get_follower_score(beam):
    listener_score = sum(map(float, beam["listener_scores"])) / len(
        beam["listener_scores"]
    )
    return listener_score


def combine_scores(spk, flw, vln_bert, alpha, beta, scale):
    spk_flw = beta * spk + (1 - beta) * flw
    combined = alpha * scale * vln_bert + (1 - alpha) * spk_flw
    return combined


def get_success_rate(
    speaker_scores, follower_scores, vln_bert_scores, errors, alpha, beta, scale
):
    sr = []
    for spk, flw, vln_bert, err in zip(
        speaker_scores, follower_scores, vln_bert_scores, errors
    ):
        scores = combine_scores(spk, flw, vln_bert, alpha, beta, scale)
        idx = np.argmax(scores)
        if err[idx] < 3.0:
            sr.append(1.0)
        else:
            sr.append(0.0)
    return 100.0 * np.mean(sr)


def main():
    args = parse_args()

    instr_id_to_goal, instr_id_to_scan = load_vln_data(args.split)
    instr_id_to_scores = load_results_data(args.scores)
    instr_id_to_beams, instr_id_to_exploration_path = load_beamsearch_data(
        args.beam_scores
    )

    scans = set(instr_id_to_scan.values())
    dist = load_distances(scans)

    speaker_scores, follower_scores, vln_bert_scores, errors = [], [], [], []
    for instr_id in instr_id_to_scores:
        goal = instr_id_to_goal[instr_id]
        scan = instr_id_to_scan[instr_id]
        beams = instr_id_to_beams[instr_id]

        # scores
        speaker_scores.append(np.array([get_speaker_score(beam) for beam in beams]))
        follower_scores.append(np.array([get_follower_score(beam) for beam in beams]))
        vln_bert_scores.append(np.array(instr_id_to_scores[instr_id]))

        # errors
        stops = [beam["trajectory"][-1][0] for beam in beams]
        errors.append([dist[scan][stop][goal] for stop in stops])

    # grid search
    best = {"alpha": None, "beta": None, "scale": None, "sr": -np.inf}
    for ii, alpha in enumerate(np.linspace(0, 1, 101)):
        for beta in np.linspace(0, 1, 101):
            for scale in [1e-1, 1e-2]:
                sr = get_success_rate(
                    speaker_scores,
                    follower_scores,
                    vln_bert_scores,
                    errors,
                    alpha,
                    beta,
                    scale,
                )
                if sr > best["sr"]:
                    best = {"alpha": alpha, "beta": beta, "scale": scale, "sr": sr}

        best_string = {k: f"{best[k]:0.2f}" for k in best}
        print(f"[{ii:03d}] best:{best_string}")


if __name__ == "__main__":
    sys.exit(main())
