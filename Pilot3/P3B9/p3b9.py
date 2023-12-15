import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {
        "name": "per_device_train_batch_size",
        "type": int,
        "default": 16,
        "help": "Batch per device in training",
    },
    {
        "name": "gradient_accumulation_steps",
        "type": int,
        "default": 1,
        "help": "Number of steps for accumulating gradient",
    },
    {
        "name": "max_len",
        "type": int,
        "default": 512,
        "help": "Max length for truncation",
    },
    {
        "name": "weight_decay",
        "type": float,
        "default": 0.0000,
        "help": "ADAM weight decay",
    },
    {
        "name": "adam_beta1",
        "type": float,
        "default": 0.9,
        "help": "ADAM beta1 parameter",
    },
    {
        "name": "adam_beta2",
        "type": float,
        "default": 0.98,
        "help": "ADAM beta2 parameter",
    },
    {
        "name": "adam_epsilon",
        "type": float,
        "default": 2e-8,
        "help": "ADAM epsilon parameter",
    },
    {"name": "max_steps", "type": int, "default": 10, "help": "Max training steps"},
    {"name": "warmup_steps", "type": int, "default": 1, "help": "Warmup steps"},
    {
        "name": "model_name_or_path",
        "type": str,
        "default": None,
        "help": "model name or path",
    },
    {
        "name": "savepath",
        "type": str,
        "default": "./pretrained",
        "help": "config path for saving",
    },
    {
        "name": "token_max_len",
        "type": int,
        "default": 128,
        "help": "Max length for tokenizer",
    },
    {
        "name": "name_pretrained_tokenizer",
        "type": str,
        "default": "pubmed_bert-vocab.txt",
        "help": "file name where pretrained tokenizer is stored",
    },
    {
        "name": "dataset",
        "type": str,
        "default": "part-000000.tar",
        "help": "file name of dataset",
    },
    {
        "name": "data_len_gpu",
        "type": int,
        "default": 1000,
        "help": "Total data len per gpu",
    },
    {
        "name": "vocab_size",
        "type": int,
        "default": 30_000,
        "help": "Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling BertModel or TFBertModel.",
    },
    {
        "name": "hidden_size",
        "type": int,
        "default": 768,
        "help": "Dimensionality of the encoder layers and the pooler layer.",
    },
    {
        "name": "intermediate_size",
        "type": int,
        "default": 3072,
        "help": "Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    },
    {
        "name": "max_position_embeddings",
        "type": int,
        "default": 512,
        "help": "The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).",
    },
    {
        "name": "num_attention_heads",
        "type": int,
        "default": 12,
        "help": "Number of attention heads for each attention layer in the Transformer encoder.",
    },
    {
        "name": "num_hidden_layers",
        "type": int,
        "default": 12,
        "help": "Number of hidden layers in the Transformer encoder.",
    },
    {
        "name": "type_vocab_size",
        "type": int,
        "default": 2,
        "help": "The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
    },
]

required = [
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_len",
    "learning_rate",
    "weight_decay",
    "adam_beta2",
    "adam_epsilon",
    "max_steps",
    "warmup_steps",
]


class BenchmarkP3B9(candle.Benchmark):
    """Benchmark for BERT"""

    def set_locals(self):
        """Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
