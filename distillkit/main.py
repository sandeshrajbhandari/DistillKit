# Copyright 2025 Arcee AI
import hashlib
import json
import logging
import os
import re
from typing import Any

import click
import datasets
import torch
import transformers
import trl
import yaml
from accelerate import Accelerator

from distillkit.compression import LogprobCompressor
from distillkit.configuration import (
    DatasetConfiguration,
    DatasetPath,
    DistillationRunConfig,
    HfRepoDataset,
    LocalDataset,
    TeacherDatasetConfig,
    TeacherModelConfig,
)
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.signals import (
    NullSignalSource,
    OfflineSignalSource,
    OnlineSignalSource,
    SignalSource,
)
from distillkit.trainer import DistillationTrainer

LOG = logging.getLogger(__name__)


def _format_row(
    example: dict[str, Any], tokenizer: transformers.PreTrainedTokenizer
) -> dict[str, Any]:
    if ("input_ids" in example) or ("text" in example):
        # either pretokenized or raw completion - no formatting needed
        return {}
    elif "conversations" in example:
        conversations = example["conversations"]

        messages = []
        for conversation in conversations:
            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
            }
            role = role_map.get(conversation.get("from", ""), None)
            if role:
                messages.append(
                    {"role": role, "content": conversation.get("value", "")}
                )

        # Apply chat template to create a single string. SFTTrainer will handle tokenization.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    elif "messages" in example:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    else:
        raise RuntimeError("Expected `text`, `messages`, or `conversations` column")


def _load_dataset(
    path: DatasetPath,
    seed: int | None,
    num_samples: int | None,
    tokenizer: transformers.PreTrainedTokenizer,
    prepared_dataset_path: str | None = None,
    keep_in_memory: bool | None = None,
    prepacked: bool = False,
    streaming: bool = False,
) -> datasets.Dataset | datasets.IterableDataset:
    if prepared_dataset_path:
        honk = json.dumps(
            {
                "path": path.model_dump(),
                "seed": seed,
                "num_samples": num_samples,
            }
        )
        logging.info(f"Dataset spec: {honk}")
        ds_hash = hashlib.sha256(honk.encode()).hexdigest()
        full_prepared_path = os.path.join(prepared_dataset_path, f"dataset-{ds_hash}")
        if os.path.exists(full_prepared_path):
            return datasets.load_from_disk(full_prepared_path)
    else:
        full_prepared_path = None
    if isinstance(path, HfRepoDataset):
        res = datasets.load_dataset(
            path.repo_id,
            name=path.config_name,
            revision=path.revision,
            split=path.split,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
        )
    elif isinstance(path, LocalDataset):
        res = datasets.load_from_disk(path.disk_path, keep_in_memory=keep_in_memory)
        if path.split:
            res = res[path.split]
        elif isinstance(res, datasets.DatasetDict):
            raise ValueError(
                "Dataset dict found but no split specified. Please specify a split."
            )
    else:
        raise ValueError(
            "Unsupported dataset type. Please provide a valid Hugging Face repo ID or local dataset path."
        )

    if streaming:
        if num_samples:
            res = res.take(num_samples)
    else:
        if prepacked:
            last_idx = len(res) - 1
            while len(res) >= 2 and len(res[last_idx]["input_ids"]) != len(
                res[0]["input_ids"]
            ):
                last_idx -= 1
            if last_idx <= 0:
                raise RuntimeError("Dataset config is probs wrong")
            res = res.select(range(last_idx + 1))

        if seed:
            res = res.shuffle(seed=seed)
        if num_samples:
            res = res.select(range(num_samples))
    if (
        (not prepacked)
        and ("text" not in res.column_names)
        and ("input_ids" not in res.column_names)
    ):
        res = res.map(
            _format_row,
            remove_columns=res.column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
    if full_prepared_path:
        os.makedirs(full_prepared_path, exist_ok=True)
        logging.info(
            f"Saving prepared dataset to {full_prepared_path} (hash: {ds_hash}, path: {path}, seed: {seed}, num_samples: {num_samples})"
        )
        res.save_to_disk(full_prepared_path)
        del res
        return datasets.load_from_disk(
            full_prepared_path, keep_in_memory=keep_in_memory
        )
    return res


def load_data(
    config: DatasetConfiguration,
    tokenizer: transformers.PreTrainedTokenizer,
    keep_in_memory: bool | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    Load the train (and optionally eval) datasets as specified in the configuration.
    """

    LOG.info(
        f"Loading datasets: {config.train_dataset} (train), {config.eval_dataset} (eval)"
    )
    ds_train = _load_dataset(
        config.train_dataset,
        config.seed,
        config.num_samples,
        tokenizer=tokenizer,
        prepared_dataset_path=config.prepared_dataset_path,
        keep_in_memory=keep_in_memory,
        prepacked=config.prepacked,
        streaming=config.streaming,
    )
    ds_eval = None
    if config.eval_dataset:
        ds_eval = _load_dataset(
            config.eval_dataset,
            config.seed,
            config.num_eval_samples,
            tokenizer=tokenizer,
            prepared_dataset_path=config.prepared_dataset_path,
            keep_in_memory=keep_in_memory,
            prepacked=config.prepacked,
            streaming=config.streaming,
        )
    return ds_train, ds_eval


def load_student_model(
    config: DistillationRunConfig,
    tokenizer_vocab_size: int,
) -> transformers.PreTrainedModel:
    if config.functionary_packing:
        monkey_patch_packing_for_model(config.train_model)
    auto_cls = getattr(transformers, config.model_auto_class, None)
    if auto_cls is None:
        raise ValueError(
            f"Model class {config.model_auto_class} not found in transformers."
        )
    LOG.info(f"Loading model {config.train_model} with class {auto_cls}")
    extra_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.use_flash_attention:
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        extra_kwargs["torch_dtype"] = torch.bfloat16
    model = auto_cls.from_pretrained(
        config.train_model,
        **extra_kwargs,
        **config.model_kwargs,
    )
    LOG.info("Loaded model.")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if (
        model_vocab_size != tokenizer_vocab_size
        or config.resize_embeddings_to_multiple_of
    ):
        model.resize_token_embeddings(
            tokenizer_vocab_size,
            pad_to_multiple_of=config.resize_embeddings_to_multiple_of,
        )
        new_model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if new_model_vocab_size != model_vocab_size:
            LOG.info(
                f"Resized model vocab size from {model_vocab_size} to {new_model_vocab_size}"
            )

    model: transformers.PreTrainedModel
    if config.lora is not None:
        from peft import LoraConfig, get_peft_model

        target_modules = config.lora.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_kwargs = {}
        if config.lora.lora_dtype is not None:
            lora_kwargs["dtype"] = config.lora.lora_dtype
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=target_modules,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            use_rslora=config.lora.use_rslora,
            **lora_kwargs,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if config.frozen_modules:
        module_set = set(config.frozen_modules)
        seen = set()
        for name, module in model.named_modules():
            if name in module_set:
                module.requires_grad_(False)
                seen.add(name)
        unseen = module_set - seen
        LOG.info(f"Froze {len(seen)} modules")
        if unseen:
            raise ValueError(f"Frozen modules not found in model: {', '.join(unseen)}")
    if config.frozen_res:
        num_frozen = 0
        frozen_res = [re.compile(s) for s in config.frozen_res]
        for name, param in model.named_parameters():
            if any(fre.search(name) for fre in frozen_res):
                param.requires_grad = False
                num_frozen += 1
        if num_frozen:
            print(f"Froze {num_frozen} tensors by regular expression")
    return model


def create_signal_source(
    config: DistillationRunConfig, vocab_size: int
) -> SignalSource:
    if config.teacher is None:
        return NullSignalSource()
    if isinstance(config.teacher, TeacherDatasetConfig):
        compressor = LogprobCompressor(
            config=config.teacher.logprob_compressor,
            legacy_config=config.teacher.legacy_logit_compression,
        )
        return OfflineSignalSource(
            compressor,
            vocab_size=vocab_size,
            sub_top_k=config.teacher.sub_top_k,
        )
    elif isinstance(config.teacher, TeacherModelConfig):
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.teacher.path, **(config.teacher.kwargs or {})
        )
        return OnlineSignalSource(
            teacher_model, vocab_size=vocab_size, sparsify_top_k=config.teacher.top_k
        )
    else:
        raise RuntimeError("Teacher configuration invalid")


def collate_packed_batch(examples):
    # all sequences in the batch already have the same length
    # so we can directly stack them
    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


def _find_common_token_ids(text: str, tokenizer) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _make_response_collator(
    base_collator, instruction_ids: list[int], response_ids: list[int]
):
    len_inst = len(instruction_ids)
    len_resp = len(response_ids)

    def _collate_fn(examples):
        batch = base_collator(examples)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)
        B, L = labels.shape

        for b in range(B):
            ids = input_ids[b].tolist()
            i = 0
            while i < L:
                if ids[i : i + len_resp] == response_ids:
                    resp_start = i + len_resp
                    j = resp_start
                    while j < L:
                        if ids[j : j + len_inst] == instruction_ids:
                            break
                        j += 1
                    labels[b, resp_start:j] = input_ids[b, resp_start:j]
                    i = j
                else:
                    i += 1

        batch["labels"] = labels
        return batch

    return _collate_fn


def load_tokenizer(config: DistillationRunConfig) -> transformers.PreTrainedTokenizer:
    if isinstance(config.teacher, TeacherModelConfig):
        src_path = config.teacher.path
        logging.info("Using teacher's tokenizer")
    else:
        src_path = config.train_model
        logging.info("Using student's tokenizer")
    return transformers.AutoTokenizer.from_pretrained(
        src_path,
        trust_remote_code=config.trust_remote_code,
    )


def do_distill(config: DistillationRunConfig, config_source: str | None = None):
    os.makedirs(config.output_path, exist_ok=True)
    if config_source is None:
        config_source = yaml.safe_dump(config.model_dump(mode="json", by_alias=True))
    with open(os.path.join(config.output_path, "distillkit_config.yaml"), "w") as f:
        f.write(config_source)

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    accelerator = Accelerator()
    with accelerator.main_process_first():
        tokenizer = load_tokenizer(config)
        ds_train, ds_eval = load_data(config.dataset, tokenizer)

        tokenizer_vocab_size = max(
            len(tokenizer.get_vocab()),
            max(tokenizer.get_vocab().values()) + 1,
        )

    model = load_student_model(config, tokenizer_vocab_size)

    config_kwargs = dict(config.training_args)
    dataset_kwargs = config_kwargs.pop("dataset_kwargs", {})
    if config.dataset.prepacked:
        dataset_kwargs["skip_prepare_dataset"] = True
    max_length = config_kwargs.pop("max_length", config.sequence_length)
    training_arguments = trl.SFTConfig(
        **config_kwargs,
        max_length=max_length,
        output_dir=config.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    signal_source = create_signal_source(config, tokenizer_vocab_size)
    if config.layer_mapping is not None:
        if not isinstance(signal_source, OnlineSignalSource):
            raise RuntimeError(
                "Hidden state distillation not supported for offline teachers"
            )
        teacher_hidden_size = signal_source.teacher_model.config.hidden_size
        if config.layer_mapping == "all":
            mapping = [(i, i) for i in range(model.config.num_hidden_layers)]
        else:
            mapping = config.layer_mapping
        hsm = HiddenStateMapping(
            student=model,
            teacher_hidden_size=teacher_hidden_size,
            layer_mapping=mapping,
            force_projection=config.force_hidden_state_projection,
        )
    else:
        hsm = None
    if config.dataset.prepacked:
        if config.dataset.response_only:
            inst_ids = _find_common_token_ids(config.dataset.instruction_part, tokenizer)
            resp_ids = _find_common_token_ids(config.dataset.response_part, tokenizer)
            data_collator = _make_response_collator(
                collate_packed_batch, inst_ids, resp_ids
            )
        else:
            data_collator = collate_packed_batch
    else:
        data_collator = None

    trainer = DistillationTrainer(
        model=model,
        config=config,
        signal_source=signal_source,
        hidden_state_mapping=hsm,
        true_vocab_size=tokenizer_vocab_size,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=training_arguments,
        data_collator=data_collator,
        processing_class=None if config.dataset.prepacked else tokenizer,
    )

    resume_from_checkpoint = config.training_args.get("resume_from_checkpoint", None)

    LOG.info("Starting training.")
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    LOG.info(f"Finished training. Saving model to {config.output_path}.")
    if config.lora is not None:
        model.save_pretrained(config.output_path)
    else:
        trainer.save_model(config.output_path)
    LOG.info("Done.")


@click.command("distillkit-offline")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)
    do_distill(config)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
