## P3B9: Pretraining BERT on a small subset of PubMed abstracts

**Overview**: This benchmark uses masked language learning to pretraing a BERT model. As it is only being trained on a small dataset, the purpose of this benchmark is to evaluate the pretraining performance on new accelerator hardware.

**Files**:

1. part-000000.tar is a tar achieve with a set of 1000 PubMed abstracts and formatted for [WebDataset](https://github.com/webdataset/webdataset).
2. pubmed_bert-vocab.txt is the tokenizer vocabulary.
3. bert_webdataset.py contains the model training code, relying heavily on [Huggingface Transformers](https://github.com/huggingface/transformers)
4. run.sh is a sample runscript for running this benchmark locally.

Benchmark requires webdataset==0.1.62.
