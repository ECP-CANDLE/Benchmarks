import torch
import numpy as np
import transformers
import json
from transformers.pipelines.fill_mask import FillMaskPipeline
from typing import Optional


class MultipleMaskPipeline(FillMaskPipeline):
    """Class to handle mask replacement using a model from hugging face."""

    def __call__(self, *args, targets=None, top_k: Optional[int] = None, **kwargs):
        inputs = self.tokenizer(*args, truncation=True, padding=True, max_length=512, return_tensors='pt')
        # print(inputs)
        outputs = self._forward(inputs, return_tensors=True)

        results = []
        batch_size = outputs.size(0)

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).flatten()
            logits = outputs[i, masked_index, :]
            probs = logits.softmax(dim=-1)
            values, predictions = probs.topk(top_k if top_k is not None else self.top_k)

            possible_indices = torch.zeros(len(predictions), dtype=torch.long)

            for k in range(self.top_k):
                indices = None
                score = 0.0
                if k == 0:
                    indices = predictions[:,0]
                    score = torch.prod(values[:,0])
                else:
                    max_score = -1
                    best_index = -1
                    for j in range(len(predictions)):
                        current_indices = possible_indices.detach().clone()
                        current_indices[j] += 1
                        current_score = torch.prod(torch.gather(values, 1, current_indices.unsqueeze(1)))
                        if current_score > max_score:
                            max_score = current_score
                            best_index = j

                    possible_indices[best_index] += 1
                    indices = torch.gather(predictions, 1, possible_indices.unsqueeze(1)).flatten()
                    score = max_score

                tokens = input_ids.numpy()
                tokens[masked_index] = indices

                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens, skip_special_tokens=True),
                        "score": score.item(),
                        "tokens": indices,
                        "token_str": self.tokenizer.decode(indices),
                    }
                )

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results

class Evolution():
    """Class to mutate and recombine sequences based on masking and replacement."""

    def __init__(self, tokenizer_directory, model_directory=None, top_k=5, device=-1, max_len=-1, mutation_parameter=0.5):
        """Constructor for Evolution class.

        Note:
            Pretrained tokenizer and/or model can be generated using hugging face (https://huggingface.co/transformers/).

        Args:
            tokenizer_directory (str): Directory location for pretrained tokenizer
            model_diretory (str): Directory location for pretrined model
            top_k (int): Number of mutation options to generate
            device (int): GPU device number to use for model mask predictions
            max_len (int): Maximum length of masked sequences
            mutation_parameter (float): Probability of mask replacement

        """
        super().__init__()

        # no longer necessary - enabled with MultipleMaskPipeline
        # consistency check - currently model pipeline only supports one mask
        # if model_directory is not None:
        #     if max_mask_number != 1:
        #         raise ValueError('Error: max_mask_number must be 1 if a model_directory is provided')

        # store input parameters
        self._max_len = max_len
        self._top_k = top_k
        self._mutation_parameter = mutation_parameter

        # initialize tokenizer
        try:
            with open(tokenizer_directory + '/config.json', 'r') as f:
                tokenizer_config = json.load(f)
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, **tokenizer_config, truncation=True, max_length=512)
        except:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, truncation=True, max_length=512)
        # print(self._tokenizer.model_max_length)
        self._vocab_size = self._tokenizer.vocab_size + len(self._tokenizer.get_added_vocab())

        # intialize model
        self._model = None
        if model_directory is not None:
            self._model = transformers.AutoModelForMaskedLM.from_pretrained(model_directory)
            self._fill_mask = MultipleMaskPipeline(model=self._model, tokenizer=self._tokenizer, top_k=self._top_k, device=device)

    def _sample_token(self):
        """Sample to random token from the tokenizer vocab.

        Returns:
            str from the vocab
        """
        token_id = np.random.choice(self._vocab_size)
        return self._tokenizer.convert_ids_to_tokens(token_id)

    def _random_mask_fill(self, sequence):
        """Fill masks with random tokens
        
        Args:
            sequence (str): Sequence with mask tokens

        Returns:
            str with mask randomly filled

        """
        split_sequence = sequence.split()
        for i in range(len(split_sequence)):

            # update to handle cases with where ## combines tokens
            # if split_sequence[i] == self._tokenizer.mask_token:
            #     split_sequence[i] = self._sample_token()

            if self._tokenizer.mask_token in split_sequence[i]:
                split_sequence[i] = split_sequence[i].replace(self._tokenizer.mask_token, self._sample_token())

        return self._tokenizer.convert_tokens_to_string(split_sequence)

    def generate_mutation_masks(self, sequences, task):
        """Generate a sequence with mutations masks."

        Args:
            sequences (List[str]): List of text sequences
            task (str): type of mutation to generate (i.e. replace, insert, delete)

        Returns:
            List[str] of masked sequences

        """
        # store masked output sequences
        output = []

        for s in sequences:

            # make sure mask token is not already present
            s = s.replace(self._tokenizer.mask_token, '')

            # convert to tokesn - note this will add special tokens at beginning/end
            input_enc = self._tokenizer(s)

            # note: expected that enc_len >=2 because of special beginning/end tokens inserted by tokenizer
            enc = input_enc['input_ids']
            reverse = self._tokenizer.convert_ids_to_tokens(enc)
            enc_len = len(enc)

            # checks on encoding length
            if enc_len <= 2:
                task = 'insert'
            elif enc_len < 4 and task == 'delete':
                task = 'replace'

            # 3 options
            # insert: insert <mask> after the selected index
            # replace: replace token at selected index with <mask>
            # delete: replace token at selected index with <mask> and delete next token
            maskIndex = set()
            insertIndex = set()
            deleteIndex = set()
            mask_number = np.random.binomial(enc_len-2, self._mutation_parameter)
            mask_number = max(1, mask_number)
            maskIndex = set(np.random.choice(np.arange(1,enc_len-1), mask_number, replace=False))

            if task == 'insert':
                insertIndex.add(np.random.choice(np.arange(0, enc_len-1)))
            elif task == 'delete':
                totalIndex = set(np.arange(1,enc_len-1))
                leftOverIndex = set.difference(totalIndex, maskIndex)
                if len(leftOverIndex) > 0:
                    deleteIndex.add(np.random.choice(list(leftOverIndex)))

            # generate masked sequence
            masked_sequence = []
            if 0 in insertIndex:
                masked_sequence.append(self._tokenizer.mask_token)
            for i in range(1, enc_len - 1):                
                if i in maskIndex:
                    masked_sequence.append(self._tokenizer.mask_token)
                elif i in deleteIndex:
                    continue
                else:
                    masked_sequence.append(reverse[i])

                if i in insertIndex:
                    masked_sequence.append(self._tokenizer.mask_token)   

            # make sure sequence is not too long (2) for special tokens
            if self._max_len > 0 and len(masked_sequence) > self._max_len - 2:
                masked_sequence = masked_sequence[:self._max_len-2]
                # if mask was eliminated put it at the end
                if self._tokenizer.mask_token not in masked_sequence:
                    masked_sequence[-1] = self._tokenizer.mask_token

            output.append(self._tokenizer.convert_tokens_to_string(masked_sequence))

        return output

    def generate_recombination_masks(self, sequences_1, sequences_2):
        """Combine seqences by slicing tokens and inserting mask in between.

        Args:
            sequences_1 (List[str]): List of text sequences
            sequences_2 (List[str]): List of text sequences 

        Returns:
            List[str] of masked sequences

        """
        # store masked output sequences
        output = []

        # check that sequences have the same length
        if len(sequences_1) != len(sequences_2):
            raise ValueError('Error: sequences_1 and sequences_2 must have the same length')

        # if input1, input2 are different lengths, use minimum
        input_len = len(sequences_1)

        for i in range(input_len):

            # randomly order inputs
            string1 = sequences_1[i]
            string2 = sequences_2[i]
            random_order = np.random.rand()
            if random_order > 0.5:
                string1 = sequences_2[i]
                string2 = sequences_1[i]

            # make sure mask token not already present
            string1 = string1.replace(self._tokenizer.mask_token, '')
            string2 = string2.replace(self._tokenizer.mask_token, '')

            # generate incoding for both parents
            input1_enc = self._tokenizer(string1)
            input2_enc = self._tokenizer(string2)
            enc1_len = len(input1_enc['input_ids'])
            enc2_len = len(input2_enc['input_ids'])

            # select sequences from inputs - assume nonempty input (i.e. len >= 3)
            index1 = 2
            if enc1_len > 3:
                index1 = np.random.randint(2, enc1_len - 1)
            index2 = 1
            if enc2_len > 3:
                index2 = np.random.randint(1, enc2_len - 2)
            split1_enc = input1_enc['input_ids'][1:index1]
            split2_enc = input2_enc['input_ids'][index2:enc2_len-1]
    
            # tokens back to string
            reverse1 = self._tokenizer.convert_ids_to_tokens(split1_enc)
            reverse2 = self._tokenizer.convert_ids_to_tokens(split2_enc)

            masked_sequence = reverse1
            masked_sequence.append(self._tokenizer.mask_token)
            masked_sequence += reverse2

            # insert additional masks based on mutation parameter
            for j in range(len(masked_sequence)):
                r_num = np.random.rand()
                if r_num < self._mutation_parameter:
                    masked_sequence[j] = self._tokenizer.mask_token

            # make sure sequence is not too long (2) for special tokens
            if self._max_len > 0 and len(masked_sequence) > self._max_len - 2:
                masked_sequence = masked_sequence[:self._max_len-2]
                # if mask was eliminated put it at the end
                if self._tokenizer.mask_token not in masked_sequence:
                    masked_sequence[-1] = self._tokenizer.mask_token

            output.append(self._tokenizer.convert_tokens_to_string(masked_sequence))

        return output

    def replace_masks(self, masked_sequences):
        """Generate mutated sequences by replacing mutation masks.

        Args:
            sequences (List[str]): List of text sequences

        Returns:
            List[List[Dict[str,] of mutated sequences

        """  
         # use model or randomly fill masks
        if self._model is not None:
            batch_results = self._fill_mask(masked_sequences)
        else:
            # format batch results the same as hugging face fill_mask pipeline
            batch_results = []
            for sample in masked_sequences:
                batch_results.append([])
                for _ in range(self._top_k):
                    mutated_sequence = self._random_mask_fill(sample)
                    batch_results[-1].append({'sequence': mutated_sequence})

        # make sure formatting is consistent for list of length 1
        try:
            batch_results[0][0]
        except:
            batch_results = [batch_results]

        return batch_results

    def mutate_sequences(self, sequences, task='replace'):
        """Mutate a list of sequences.

        Args:
            sequences (List[str]): List of text sequences
            task (str): type of mutation to generate (i.e. replace, insert, delete)

        Returns:
            List[List[Dict[str,] of mutated sequences

        """
        batch_masks = self.generate_mutation_masks(sequences, task=task)
        batch_results = self.replace_masks(batch_masks)

        return batch_results

    def recombine_sequences(self, sequences_1, sequences_2):
        """Apply recombination to two lists of sequences.

        Args:
            sequences (List[str]): List of text sequences
            task (str): type of mutation to generate (i.e. replace, insert, delete)

        Returns:
            List[List[Dict[str,] of mutated sequences

        """
        batch_masks = self.generate_recombination_masks(sequences_1, sequences_2)
        batch_results = self.replace_masks(batch_masks)

        return batch_results

# some sample use cases
if __name__ == '__main__':
    print('\nExamples of using masked language model to mutate sentences', flush=True)
    print('Model: bert-base-uncased')

    # use pretrained bert tokenizer/model from hugging face, run on GPU 0
    bert_mutation_operator = Evolution(
        tokenizer_directory='bert-base-uncased', 
        model_directory='bert-base-uncased',
        top_k=5,
        device=0,
        max_len=512,
        mutation_parameter=0.5)

    # use pretrained bert tokenizer from hugging face, random mask replacement
    random_operator = Evolution(
        tokenizer_directory='bert-base-uncased', 
        top_k=5,
        device=0,
        max_len=512,
        mutation_parameter=0.5)

    print('Mutation Example')
    sample_sentence = 'This is going to be a great [MASK] day'
    print('\nSample sentence:', sample_sentence, end='\n\n')

    print('BERT Suggestions to Fill Mask')
    sample = bert_mutation_operator.replace_masks([sample_sentence])
    for results in sample:
        for r in results:
            print(r['sequence'], r['score'])
    print()

    print('Random Suggestions to Fill Mask')
    sample = random_operator.replace_masks([sample_sentence])
    for results in sample:
        for r in results:
            # note that no score is provided for random mutations
            print(r['sequence'])
    print()

    print('Recombination Example')
    sample_sentence_1 = 'This is going to be a great day'
    sample_sentence_2 = 'The weather is cloudy outside'
    print('\nSample sentence 1:', sample_sentence_1)
    print('Sample sentence 2:', sample_sentence_2)
    masked_combination = bert_mutation_operator.generate_recombination_masks([sample_sentence_1], [sample_sentence_2])
    print('Sample mask:', masked_combination[0], end='\n\n')

    print('BERT Suggestions to Fill Mask')
    sample = bert_mutation_operator.replace_masks(masked_combination)
    for results in sample:
        for r in results:
            print(r['sequence'], r['score'])
    print()

    print('Random Suggestions to Fill Mask')
    sample = random_operator.replace_masks(masked_combination)
    for results in sample:
        for r in results:
            # note that no score is provided for random mutations
            print(r['sequence'])
    print()