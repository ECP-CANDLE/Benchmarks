import candle
import p3b9 as bmk
import torch
import webdataset as wd
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


class truncate(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, doc):
        return doc[: self.max_len]


def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""

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

    # Repair argument overlap by duplication
    # (i.e. arguments that have the same function
    # but slightly different names between CANDLE
    # and hf parsers are copied from CANDLE keyword
    # to expected hf keyword)
    args["seed"] = args["rng_seed"]
    if "train_bool" in args.keys():
        args["do_train"] = args["train_bool"]
    if "eval_bool" in args.keys():
        args["do_eval"] = args["eval_bool"]
    if "epochs" in args.keys():
        args["num_train_epochs"] = args["epochs"]

    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(args)[0]
    print(training_args)

    # Convert from dictionary to structure
    args = candle.ArgumentStruct(**args)

    trunc = truncate(args.max_len)

    print("total data len per gpu:", args.data_len_gpu)
    dataset = (
        wd.Dataset(args.dataset, length=args.data_len_gpu, shuffle=True)
        .decode("torch")
        .rename(input_ids="pth")
        .map_dict(input_ids=trunc)
        .shuffle(1000)
    )
    tokenizer = BertTokenizer.from_pretrained(args.name_pretrained_tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    config = BertConfig(
        vocab_size=args.vocab_size,
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
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.savepath)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
