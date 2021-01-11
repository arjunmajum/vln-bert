import argparse
import json
import os
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.63,
        help="alpha value from grid search (default: 0.63)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.94,
        help="beta value from grid search (default: 0.94)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1e-2,
        help="scale value from grid search (default: 1e-2)",
    )
    parser.add_argument(
        "--add-exploration-path",
        action="store_true",
        help="add exploration path for leaderboard evaluation",
    )
    # fmt: on
    return parser.parse_args()


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


def main():
    args = parse_args()

    instr_id_to_scores = load_results_data(args.scores)
    instr_id_to_beams, instr_id_to_exploration_path = load_beamsearch_data(
        args.beam_scores
    )

    results = []
    for instr_id in instr_id_to_scores:
        beams = instr_id_to_beams[instr_id]
        speaker_scores = np.array([get_speaker_score(beam) for beam in beams])
        follower_scores = np.array([get_follower_score(beam) for beam in beams])

        vln_bert_scores = np.array(instr_id_to_scores[instr_id])

        # combine scores
        scores = combine_scores(
            speaker_scores,
            follower_scores,
            vln_bert_scores,
            args.alpha,
            args.beta,
            args.scale,
        )

        # get best path
        idx = np.argmax(scores)
        trajectory = beams[idx]["trajectory"]

        if args.add_exploration_path:
            exploration_path = [
                (viewpoint, 0.0, 0.0)
                for viewpoint in instr_id_to_exploration_path[instr_id]
            ]
            trajectory = exploration_path + trajectory

        results.append(
            {
                "instr_id": instr_id,
                "trajectory": trajectory,
            }
        )

    output_path = os.path.join(
        os.path.dirname(args.scores), "combined_" + os.path.basename(args.scores)
    )
    json.dump(results, open(output_path, "w"))

    print(f"saved combined results to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
