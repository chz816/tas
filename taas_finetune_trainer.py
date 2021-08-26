#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from model import TAASForConditionalGeneration
from topic_models.data_preparation import TopicModelDataPreparation
from topic_models.preprocessing import WhiteSpacePreprocessing

from taas_seq2seq_trainer import Seq2SeqTrainer
from seq2seq_training_args import Seq2SeqTrainingArguments

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from utils import (
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
    TAASSeq2SeqDataset,
    TAASSeq2SeqDataCollator,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_decoder: bool = field(default=False, metadata={"help": "Whether tp freeze the decoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    topic_num: int = field(default=1024, metadata={"help": "Number of topics learned in NTM"})
    loss_alpha: float = field(default=0, metadata={"help": "Weight for calculate total loss"})
    loss_beta: float = field(default=1, metadata={"help": "Weight for calculate total loss"})
    topic_vocab: int = field(default=2000, metadata={"help": "Size of vocab in NTM"})

    continue_trainer: bool = field(default=False, metadata={"help": "flag variable to continue training"})
    continue_trainer_path: str = field(default=None, metadata={"help": "checkpoint path to continue training"})

    load_checkpoint: bool = field(default=False, metadata={"help": "flag variable to continue training"})
    load_checkpoint_path: str = field(default=None, metadata={"help": "checkpoint path to continue training"})

    topic_model_type: str = field(default='prodLDA', metadata={"help": "what type of topic model is employed"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. "
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if not model_args.load_checkpoint:
        model = TAASForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            topic_num=model_args.topic_num,
        )
    else:
        model = TAASForConditionalGeneration.from_pretrained(model_args.load_checkpoint_path,
                                                             topic_num=model_args.topic_num,
                                                             model_type=model_args.topic_model_type)

    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        assert (
                data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]

    if model_args.freeze_embeds:
        freeze_embeds(model)
        logger.info("Freeze the word embedding")
    # freeze encoder
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())
        logger.info("Freeze the encoder")
    # freeze decoder
    if model_args.freeze_decoder:
        freeze_params(model.get_decoder())
        assert_all_frozen(model.get_decoder())
        logger.info(f"Freeze the decoder")

    # preprocess for the topic modeling
    logger.info("Preprocess the documents for topic modeling")
    # load the input documents for training
    train_documents = [line.strip() for line in
                       open(f"{data_args.data_dir}/train.source", encoding="utf-8").readlines()]
    # load the input documents for validation
    val_documents = [line.strip() for line in open(f"{data_args.data_dir}/val.source", encoding="utf-8").readlines()]
    # load the input documents for testing set
    test_documents = [line.strip() for line in open(f"{data_args.data_dir}/test.source", encoding="utf-8").readlines()]

    # load the vocabulary - we use the training set
    sp = WhiteSpacePreprocessing(train_documents, "english")
    preprocessed_documents_for_bow_training, topic_vocab = sp.preprocess()
    # sort the topic_vocab
    topic_vocab = sorted(topic_vocab)
    # for validation set
    sp = WhiteSpacePreprocessing(val_documents, "english")
    # use the pre-defined topic_vocab, which is learned from the training set
    preprocessed_documents_for_bow_val, val_topic_vocab = sp.preprocess(vocabulary=topic_vocab)
    # for testing set
    sp = WhiteSpacePreprocessing(test_documents, "english")
    # use the pre-defined topic_vocab, which is learned from the training set
    preprocessed_documents_for_bow_test, test_topic_vocab = sp.preprocess(vocabulary=topic_vocab)
    qt = TopicModelDataPreparation(topic_vocab)
    # get bow representation for training
    bow_dataset_train = qt.create_training_set(preprocessed_documents_for_bow_training)
    # get bow representation for validation
    bow_dataset_val = qt.create_training_set(preprocessed_documents_for_bow_val)
    # get bow representation for testing
    bow_dataset_test = qt.create_training_set(preprocessed_documents_for_bow_test)
    # get the mapping between id and selected word
    id2token = bow_dataset_train.idx2token

    # Get datasets
    train_dataset = (
        TAASSeq2SeqDataset(
            tokenizer,
            type_path="train",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            bow_representation=bow_dataset_train.X,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TAASSeq2SeqDataset(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            bow_representation=bow_dataset_val.X,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        TAASSeq2SeqDataset(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            bow_representation=bow_dataset_test.X,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else None
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TAASSeq2SeqDataCollator(tokenizer, data_args, training_args.tpu_num_cores),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        topic_num=model_args.topic_num,
        alpha=model_args.loss_alpha,
        beta=model_args.loss_beta,
        id2token=id2token,
        topic_vocab=topic_vocab,
        topic_vocab_size=len(topic_vocab),
    )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        if not model_args.continue_trainer:
            train_result = trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        else:
            train_result = trainer.train(
                model_path=model_args.continue_trainer_path if os.path.isdir(model_args.continue_trainer_path) else None
            )

        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            tokenizer.save_pretrained(training_args.output_dir)

    return all_metrics


if __name__ == "__main__":
    main()
