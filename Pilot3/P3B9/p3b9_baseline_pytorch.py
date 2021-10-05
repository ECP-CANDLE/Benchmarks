import torch
import p3b9 as bmk
import candle

from dataclasses import dataclass, field
from transformers import (
BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, HfArgumentParser
)

import webdataset as wd

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


def run(args):

    # Repair argument overlap (i.e. arguments
    # that have same function but slightly different
    # names between CANDLE and hf parsers)
    args['seed'] = args['rng_seed']
    args['do_train'] = args['train_bool']

    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(args)
    training_args = training_args[0]
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
