import torch
from scoring_interface import ScoringInterface
import scipy.stats
import numpy as np

from cnn import generate_torch_data
from cnn import tokenize_data
import cnn

###############################################################################
# Class for CNN scoring
###############################################################################


class DocumentScoring(ScoringInterface):

    def __init__(self, scoring_names, selection_names, scoring_parameters={}, data_column_name='document', fitness_column_name='fitness', fitness_function=scipy.stats.hmean):
        super().__init__()

        self._name_to_function = {
            'target_probability': self._target_probability,
            'target_distance': self._target_distance,
            'predicted_class': self._predicted_class,
            'niche_index': self._niche_index_with_default,
            'target_score': self._target_score,
            'max_score': self._max_score,
            'min_score': self._min_score,
            'positive_classes': self._positive_classes
        }

        # store column name for data
        self._data_column_name = data_column_name

        # special setting
        self._target_prediction = None

        # fitness function information
        self._fitness_function = fitness_function
        self._fitness_column_name = fitness_column_name

        # genertic parameters needed for all scoring functions
        self._device = None
        self._saved_model = None
        self._data_prefix = None
        if 'device' not in scoring_parameters:
            raise KeyError('Error: device not in scoring_parameters, needed to calculate target metric')
        device = scoring_parameters['device']
        self._device = torch.device(device)
        if 'saved_model' not in scoring_parameters:
            raise KeyError('Error: saved_model not in scoring_parameters, needed to calculate target metric')
        self._saved_model = scoring_parameters['saved_model']
        if 'data_prefix' not in scoring_parameters:
            raise KeyError('Error: data_prefix not in scoring_parameters, needed to calculate target metric')
        self._data_prefix = scoring_parameters['data_prefix']

        # vocab used in model
        number_tokens = 1
        self.vocab = {}
        with open(self._data_prefix + '_vocab.txt', 'r') as f:
            for row in f:
                number_tokens += 1
                self.vocab[row.split()[0]] = int(row.split()[1])

        # classes in model
        number_classes = 0
        with open(self._data_prefix + '_labels.txt','r') as f:
            for row in f:
                number_classes += 1

        # setup model
        self._number_classes = number_classes
        self._model = cnn.Model(number_classes, cnn.Hparams(vocab_size=number_tokens))
        checkpoint = torch.load(self._saved_model, map_location='cpu')
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self._device)
        self._model.eval()

        # setup for target
        self._target_class= None
        if 'target_probability' in scoring_names or 'target_distance' in scoring_names:
            if 'target_class' not in scoring_parameters:
                raise KeyError('Error: target_class not in scoring_parameters, needed to calculate target metric')
            self._target_class = [int(x) for x in scoring_parameters['target_class'].split(',')]

        # store names of functions to be used in scoring - make sure that selection names are subset
        self._scoring_names = scoring_names
        for name in selection_names:
            if name not in self._scoring_names:
                self._scoring_names.append(name)

        # store selection names and check that they are subset of scoring names
        self._selection_names = selection_names

        # check whether scoring names are in self._name_to_function
        for name in scoring_names:
            if name not in self._name_to_function:
                raise KeyError('Error: %s not an implemented scoring function. Options are %s' %(name, ' '.join(self._name_to_function.keys())))

        # scoring to see if sentence is reasonable
        # self._bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        # self._bert_model = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(self._device)

    def prepare_data_for_scoring(self, document):
        # check for words repeated more than 2 times
        words = document.split()
        words_in_vocab = []
        word_dict = {}
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
            # if word_dict[word] > 2:
            #     return None

            if word in self.vocab:
                words_in_vocab.append(word)

        if len(words_in_vocab) < 1:
            return None

        return ' '.join(words_in_vocab)

    # generate canonical string for comparisons
    def make_canonical(self, document):

        return document

    @property
    def column_names(self):
        all_names = [self._data_column_name] + list(self._scoring_names) + [self._fitness_column_name]
        return all_names

    @property
    def selection_names(self):
        return self._selection_names

    @property
    def data_column_name(self):
        return self._data_column_name

    @property
    def fitness_column_name(self):
        return self._fitness_column_name

    # generate tokens from txt using vocab
    def txt_to_tokens(self, documents):
        return tokenize_data(documents, self.vocab)

    # use lookup to return dictionary with document scores
    def generate_scores(self, documents, scoring_names=None, batch_size=32):

        score_dict = {}

        used_names = scoring_names if scoring_names is not None else self._scoring_names
        for name in used_names:
            score_dict[name] = []

        tokenized_documents = tokenize_data(documents, self.vocab)
        with torch.no_grad():
            for k in range(0, len(tokenized_documents), batch_size):
                start_index = k
                end_index = min(len(tokenized_documents), k+batch_size)
                indices = np.arange(start_index, end_index)
                tokensBatch = generate_torch_data(tokenized_documents, indices)
                tokens_device = tokensBatch.to(self._device)
                output = self._model(tokens_device)

                # output is logits, sigmoid for probabilities for each class
                output = torch.sigmoid(output).detach().cpu().numpy()

                for name in used_names:
                    score_dict[name] += self._name_to_function[name](output)

        score_dict[self.data_column_name] = documents

        # return fitness
        score_dict['fitness'] = []
        for i in range(len(score_dict[self.data_column_name])):
            fitness_array = [score_dict[x][i] for x in self._selection_names]
            if len(fitness_array) > 0:
                score_dict['fitness'].append(self._fitness_function(fitness_array))
            else:
                score_dict['fitness'].append(0.0)

        return score_dict

    # probability for target class
    def _target_probability(self, model_output):
        return list(np.mean(model_output[:,self._target_class], axis=-1))

    # distance from target class
    def _target_distance(self, model_output):
        target = np.zeros(model_output.shape[1])
        for t in self._target_class:
            target[t] = 1
        return list(1.0 - np.linalg.norm(target - model_output, axis=1)/self._number_classes)

    def _predicted_class(self, model_output):
        return list(np.argmax(model_output, axis=1))

    def _niche_index_with_default(self, model_output, default=0):
        output = []
        predictions = (1*(model_output >= 0.5))

        for i in range(len(predictions)):
            try:
                niche_index = int(''.join([str(x) for x in predictions[i]]),2)
                output.append(niche_index)
            except:
                output.append(default)

        return output

    def _positive_classes(self, model_output):
        output = []
        predictions = (1*(model_output >= 0.5))

        for i in range(len(predictions)):
            positives = np.nonzero(predictions[i])[0].tolist()
            output.append(','.join([str(x) for x in positives]))

        return output

    def _max_score(self, model_output):
        return np.max(model_output, axis=-1).tolist()

    def _min_score(self, model_output, threshold=132):
        output = []
        for i in range(len(model_output)):
            total_positives = np.sum(model_output[i]>=0.5)
            if (total_positives == 0) or (total_positives > threshold):
                output.append(0.0)
            else:
                output.append(np.min(model_output[i] + (1.0*(model_output[i] < 0.5)), axis=-1))

        return output

    def _target_score(self, model_output, threshold=132):
        output = []
        predictions = (1.0*(model_output >= 0.5))

        for i in range(len(predictions)):
            score = 0.0

            p = predictions[i]
            if self._target_prediction is not None:
                p = self._target_pediction
                # print(p)

            # sets score to 0 for predictions with more than threshold positive classes
            total_positives = np.sum(p)
            if (total_positives <= threshold) and (total_positives > 0):
                score = 1.0 - np.linalg.norm(p  - model_output[i])/np.sqrt(self._number_classes)
            output.append(score)

        return output


# some sample use cases
if __name__ == '__main__':
    print('Examples of using molecule scoring\n', flush=True)

    examples_documents = ['This is a cocoa related cocoa document icco', 'This is a merger related acquisition sell stake sentence', 'trade of flood control products official price index at east market releases of cattle crowns in cattle trade']
    scoring_parameters = {
        "device":"cuda:0",
        "saved_model":"./pretrained/mtcnn_model_val_min.tar",
        "data_prefix":"./data/r52",
        "target_class":0
    }
    document_scoring = DocumentScoring(['target_probability', 'target_distance', 'predicted_class'], ['target_probability'], scoring_parameters=scoring_parameters)
    scores = document_scoring.generate_scores(examples_documents)
    for i in range(len(examples_documents)):
        print(examples_documents[i], scores['target_probability'][i], scores['target_distance'][i], scores['predicted_class'][i])
