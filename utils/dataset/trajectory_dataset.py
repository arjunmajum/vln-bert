# pylint: disable=no-member, not-callable
import os

import networkx as nx
import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import Dataset

from utils.dataset.common import (
    get_headings,
    get_viewpoints,
    load_distances,
    load_json_data,
    load_nav_graphs,
    randomize_tokens,
    save_json_data,
    tokenize,
)
from utils.dataset.pano_features_reader import PanoFeaturesReader


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        vln_path: str,
        tokenizer: BertTokenizer,
        pano_features_reader: PanoFeaturesReader,
        max_instruction_length: int,
        max_path_length: int,
        max_num_boxes: int,
        **kwargs,
    ):
        # load and tokenize data (with caching)
        tokenized_path = f"_tokenized_{max_instruction_length}".join(
            os.path.splitext(vln_path)
        )
        if os.path.exists(tokenized_path):
            self._data = load_json_data(tokenized_path)
        else:
            self._data = load_json_data(vln_path)
            tokenize(self._data, tokenizer, max_instruction_length)
            save_json_data(self._data, tokenized_path)

        # map path ids to indices
        self._index_to_data = []
        for i, item in enumerate(self._data):
            for j in range(len(item["instructions"])):
                self._index_to_data.append((i, j))

        # load navigation graphs
        scan_list = [item["scan"] for item in self._data]
        self._graphs = load_nav_graphs(scan_list)
        self._distances = load_distances(scan_list)

        # get all of the viewpoints for this dataset
        self._viewpoints = get_viewpoints(
            self._data, self._graphs, pano_features_reader
        )

        self._pano_features_reader = pano_features_reader
        self._max_instruction_length = max_instruction_length
        self._max_path_length = max_path_length
        self._max_num_boxes = max_num_boxes

    def __len__(self):
        return len(self._index_to_data)

    def __getitem__(self, index):
        # get indices
        data_index, instruction_index = self._index_to_data[index]
        scan_id = self._data[data_index]["scan"]
        heading = self._data[data_index]["heading"]

        # get the ground truth path features
        gt_path = self._data[data_index]["path"]
        gt_features, gt_boxes, gt_masks = self._get_path_features(
            scan_id, gt_path, heading
        )

        # get the ground truth instruction data
        gt_instr_tokens = self._data[data_index]["instruction_tokens"][
            instruction_index
        ]
        gt_instr_mask = self._data[data_index]["instruction_token_masks"][
            instruction_index
        ]
        gt_segment_ids = self._data[data_index]["instruction_segment_ids"][
            instruction_index
        ]

        # Negative 1: swap instructions
        lang_path = gt_path[:]
        lang_features, lang_boxes, lang_masks = self._get_path_features(
            scan_id, lang_path, heading
        )

        # TODO: should these be from the same scan?
        lang_index = np.random.randint(len(self._data))
        lang_data_index, lang_instruction_index = self._index_to_data[lang_index]
        lang_instr_tokens = self._data[lang_data_index]["instruction_tokens"][
            lang_instruction_index
        ]
        lang_instr_mask = self._data[lang_data_index]["instruction_token_masks"][
            lang_instruction_index
        ]
        lang_segment_ids = self._data[lang_data_index]["instruction_segment_ids"][
            lang_instruction_index
        ]

        # Negative 2: hard alternative path
        # easy_path = self._get_easy_negative_path(scan_id, gt_path)
        easy_path = self._get_hard_negative_path(scan_id, gt_path)
        if easy_path is None:
            easy_path = self._get_backup_negative_path(scan_id, gt_path)
        easy_features, easy_boxes, easy_masks = self._get_path_features(
            scan_id, easy_path, heading
        )
        easy_instr_tokens = gt_instr_tokens[:]
        easy_instr_mask = gt_instr_mask[:]
        easy_segment_ids = gt_segment_ids[:]

        # Negative 3: hard alternative path
        hard_path = self._get_hard_negative_path(scan_id, gt_path)
        if hard_path is None:
            hard_path = self._get_backup_negative_path(scan_id, gt_path)
        hard_features, hard_boxes, hard_masks = self._get_path_features(
            scan_id, hard_path, heading
        )
        hard_instr_tokens = gt_instr_tokens[:]
        hard_instr_mask = gt_instr_mask[:]
        hard_segment_ids = gt_segment_ids[:]

        # convert data into tensors
        image_features = torch.tensor(
            [gt_features, lang_features, easy_features, hard_features]
        ).float()
        image_boxes = torch.tensor(
            [gt_boxes, lang_boxes, easy_boxes, hard_boxes]
        ).float()
        image_masks = torch.tensor(
            [gt_masks, lang_masks, easy_masks, hard_masks]
        ).long()
        instr_tokens = torch.tensor(
            [gt_instr_tokens, lang_instr_tokens, easy_instr_tokens, hard_instr_tokens]
        ).long()
        instr_mask = torch.tensor(
            [gt_instr_mask, lang_instr_mask, easy_instr_mask, hard_instr_mask]
        ).long()
        segment_ids = torch.tensor(
            [gt_segment_ids, lang_segment_ids, easy_segment_ids, hard_segment_ids]
        ).long()

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self._tokenizer
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # set target
        target = torch.tensor(0).long()

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self._max_path_length * self._max_num_boxes, self._max_instruction_length
        ).long()
        path_id = torch.tensor(self._data[data_index]["path_id"])

        return (
            target,
            image_features,
            image_boxes,
            image_masks,
            instr_tokens,
            instr_mask,
            instr_targets,
            segment_ids,
            co_attention_mask,
            path_id,
        )

    def _get_easy_negative_path(self, scan_id, path):
        """ Create a negative path from the source to a random neighbor."""
        g, d = self._graphs[scan_id], self._distances[scan_id]
        source, goal = path[0], path[-1]

        # get valid neighbors within 4 and 6 hops and greater than 3m from the goal
        max_hops, min_hops, min_distance = 6, 4, 3
        neighbors = nx.single_source_shortest_path_length(g, source, cutoff=max_hops)
        neighbors = [k for k, v in neighbors.items() if v >= min_hops]
        valid = [node for node in neighbors if d[goal][node] > min_distance]
        if len(valid) == 0:
            return

        # return the shortest path to a random negative target viewpoint
        negative = np.random.choice(valid)
        return nx.dijkstra_path(g, source, negative)

    def _get_hard_negative_path(self, scan_id, path):
        """ Create a negative path that starts along the path then goes to a random neighbor."""
        g, d = self._graphs[scan_id], self._distances[scan_id]
        offset = np.random.randint(1, len(path) - 1)
        source, goal = path[offset], path[-1]

        # get valid neighbors within 4 and 6 hops and greater than 3m from the goal
        max_hops, min_hops, min_distance = 6 - offset, 4 - offset, 3
        neighbors = nx.single_source_shortest_path_length(g, source, cutoff=max_hops)
        neighbors = [k for k, v in neighbors.items() if v >= min_hops]
        valid = [node for node in neighbors if d[goal][node] > min_distance]
        if len(valid) == 0:
            return

        # return the shortest path to a random negative target viewpoint
        negative = np.random.choice(valid)
        return path[:offset] + nx.dijkstra_path(g, source, negative)

    def _get_backup_negative_path(self, scan_id, path):
        """ Create a negative path by swapping one of the viewpoints randomly. """
        negative_path = path[:]  # copy path

        swap_index = np.random.randint(len(negative_path))
        swap_image_id = np.random.choice(list(self._viewpoints[scan_id] - set(path)))
        negative_path[swap_index] = swap_image_id
        return negative_path

    # TODO: move to utils
    def _get_path_features(self, scan_id, path, first_heading):
        """ Get features for a given path. """
        headings = get_headings(self._graphs[scan_id], path, first_heading)
        # for next headings duplicate the last
        next_headings = headings[1:] + [headings[-1]]

        path_length = min(len(path), self._max_path_length)
        path_features, path_boxes, path_masks = [], [], []
        for path_idx, path_id in enumerate(path[:path_length]):
            key = scan_id + "-" + path_id

            # get image features
            features, boxes = self._pano_features_reader[
                key.encode(), headings[path_idx], next_headings[path_idx],
            ]
            num_boxes = min(len(boxes), self._max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx

            box_pad_length = self._max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self._max_path_length):
            pad_features = np.zeros((self._max_num_boxes, 2048))
            pad_boxes = np.zeros((self._max_num_boxes, 12))
            pad_boxes[:, 11] = np.ones(self._max_num_boxes) * path_idx
            pad_masks = [0] * self._max_num_boxes

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_masks.append(pad_masks)

        return np.vstack(path_features), np.vstack(path_boxes), np.hstack(path_masks)
