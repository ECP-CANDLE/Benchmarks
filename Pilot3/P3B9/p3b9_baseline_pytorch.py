import torch
import p3b9 as bmk
import candle

from dataclasses import dataclass, field
from transformers import (
BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, HfArgumentParser
)

import webdataset as wd

@dataclass
class TrainingArguments:
    """
    Arguments pertaining model training
    """
    model_name_or_path: str = field(
        default=None
    )
    savepath: str = field(
        default="./pretrained"
    )
    max_len: int = field(
        default=128
    )

training_args  output_dir=./outputs, overwrite_output_dir=False, do_train=True, do_eval=None, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=0.0001, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.98, adam_epsilon=2e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=10, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=1, logging_dir=runs/Oct05_09-42-26_cn194, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=./outputs, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=0, mp_parameters=

class truncate(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, doc):
        return doc[:self.max_len]


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b9_bench = bmk.BenchmarkP3B9(
        bmk.file_path,
        "default_model.txt",
        "pytorch",
        prog="p3b9",
        desc="BERT",
    )

    gParameters = candle.finalize_parameters(p3b9_bench)
    return gParameters


def subselect_args(list_select, args):

    sel = {}
    for ar in list_select:
        sel[ar] = args[ar]

    return sel

def run(args):


    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(args)
    args = candle.ArgumentStruct(**args)

    #parser = HfArgumentParser((ModelArguments, TrainingArguments))
    #model_args, training_args = parser.parse_args_into_dataclasses()
    #print(model_args, training_args)
    print(training_args)
    trunc = truncate(args.max_len)

    print('total data len per gpu:', args.data_len_gpu)
    dataset = wd.Dataset(args.dataset,
                     length=args.data_len_gpu, shuffle=True).decode('torch').rename(input_ids='pth').map_dict(input_ids=trunc).shuffle(1000)
    tokenizer = BertTokenizer.from_pretrained(args.name_pretrained_tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    config = BertConfig(
        vocab_size=30_000,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
    )

    if args.model_name_or_path is not None:
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    else:
        model = BertForMaskedLM(config=config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(args.savepath)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
