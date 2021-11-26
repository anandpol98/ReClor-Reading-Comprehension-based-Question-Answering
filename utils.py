import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, example_id, question, contexts, endings, label=None):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class ReclorProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] 
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context], 
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label
                    )
                )  
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending
    
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))
        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))
    return features

processors = {"reclor": ReclorProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"reclor": 4}