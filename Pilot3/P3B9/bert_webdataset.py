from dataclasses import dataclass, field
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import HfArgumentParser
import webdataset as wd


@dataclass
class ModelArguments:
    """
    Arguments pertaining tomodel/config/tokenizer
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


class truncate(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, doc):
        return doc[:self.max_len]


parser = HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
print(model_args, training_args)
trunc = truncate(model_args.max_len)

data_len = 1000
print('total data len per gpu:', data_len)
dataset = wd.Dataset('part-000000.tar',
                     length=data_len, shuffle=True).decode('torch').rename(input_ids='pth').map_dict(input_ids=trunc).shuffle(1000)

tokenizer = BertTokenizer.from_pretrained('pubmed_bert-vocab.txt')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

config = BertConfig(
    vocab_size=30_000,
    hidden_size=768,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2,
)

if model_args.model_name_or_path is not None:
    model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
else:
    model = BertForMaskedLM(config=config)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(model_args.savepath)
