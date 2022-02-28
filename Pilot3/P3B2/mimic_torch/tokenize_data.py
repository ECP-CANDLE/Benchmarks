import numpy as np
import argparse
import pandas as pd

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type=str, default='./data/notes_diagnoses_procedures.csv', help='location for input csv')
parser.add_argument('--min_token_count', type=int, default=5, help='minimum count for token to remain in vocab')
parser.add_argument('--min_class_count', type=int, default=1000, help='minimum count for token to remain in vocab')
parser.add_argument('--output_prefix', type=str, default='./data/mimic')
parser.add_argument('--val_split', type=float, default=0.1, help='fraction to randomly split for val')
parser.add_argument('--test_split', type=float, default=0.1, help='fraction to randomly split for test')
parser.add_argument('--max_document_length', type=int, default=5000, help='max length for a document')

args = parser.parse_args()

print(args)

##########################################################################################
# Helper Function
##########################################################################################

def generate_counts(input, delimiter=' ',output_interval=1000):
    input_counts = {}
    counter = 0
    for row in input:
        for elem in set(str(row).split(delimiter)):
            if elem not in input_counts:
                input_counts[elem] = 1
            else:
                input_counts[elem] += 1
        counter += 1
        if counter % output_interval == 0:
            print('Processed: ', counter, flush=True)
    
    return input_counts

def write_dict_to_file(counts_dict, output_file, threshold=0, counter_start=0):
    elems = list(counts_dict.keys())
    counts = [counts_dict[x] for x in elems]
    ordered = np.argsort(counts)[::-1]
    output = {}

    counter = counter_start
    with open(output_file, 'w') as f:
        for i in ordered:
            if counts[i] >= threshold:
                f.write('%s\t%d\t%d\n' % (elems[i], counter, counts[i]))
                output[elems[i]] = counter
                counter += 1
            else:
                break

    return output

def write_samples_to_file(samples_tokens, samples_labels, tokens_dict, labels_dict, file_tokens, file_labels, output_interval=1000):
    skipped_counter = 0
    counter = 0
    with open(file_tokens, 'w') as f:
        with open(file_labels, 'w') as g:
            for i in range(len(samples_tokens)):
                # convert text to token ids
                output_tokens = []
                for elem in samples_tokens[i]:
                    if elem in tokens_dict:
                        output_tokens.append(str(tokens_dict[elem]))
                    # else:
                    #     print(elem)

                # truncation
                if len(output_tokens) > args.max_document_length:
                    output_tokens = output_tokens[:args.max_document_length]

                # convert labels to class indices
                output_labels = []
                for l in samples_labels[i]:
                    if l in labels_dict:
                        output_labels.append(str(labels_dict[l]))

                # write converted data
                if (len(output_tokens) > 0) and (len(output_labels) > 0):
                    f.write('%s\n' % ' '.join(output_tokens))
                    g.write('%s\n' % ' '.join(output_labels))
                else:
                    skipped_counter += 1

                counter += 1
                if counter % output_interval == 0:
                    print('Converted: ', counter, flush=True)

    if skipped_counter > 0:
        print('Skipped: ', skipped_counter, flush=True)

# def write_labels_to_file(samples, labels_dict, file_name)

##########################################################################################
# Read Input Data
##########################################################################################

data = pd.read_csv(args.input_csv)

# establish train/val/test splits
total_documents = len(data)
all_indices = np.random.permutation(total_documents)
val_length = int(args.val_split*total_documents)
test_length = int(args.test_split*total_documents)
val_indices = all_indices[:val_length]
test_indices = all_indices[val_length:val_length+test_length]
train_indices = all_indices[val_length+test_length:]

sentences = data['TEXT']
train_sentences = [sentences[i].split() for i in train_indices]
val_sentences = [sentences[i].split() for i in val_indices]
test_sentences = [sentences[i].split() for i in test_indices]

labels = data['DIAG_CAT']
train_labels = [str(labels[i]).split(',') for i in train_indices]
val_labels = [str(labels[i]).split(',') for i in val_indices]
test_labels = [str(labels[i]).split(',') for i in test_indices]

print('Documents for Training: ', len(train_indices))
print('Documents for Validation: ', len(val_indices))
print('Document for Testing: ', len(test_indices), flush=True)

##########################################################################################
# Output Prefix
##########################################################################################

# Give information on min_token_count min_class_count max_document_length
parameter_tag = '_'.join([args.output_prefix, str(args.min_token_count), str(args.min_class_count), str(args.max_document_length)])
output_prefix = '%s_' % parameter_tag

##########################################################################################
# Construct Vocab
##########################################################################################

# note: this currently uses whole dataset to generate the vocab
print('Generating Vocab', flush=True)
token_counts = generate_counts(sentences)
token_dict = write_dict_to_file(token_counts, output_prefix + 'vocab.txt', args.min_token_count, 1)
print('Vocab File Written: ', output_prefix + 'vocab.txt', flush=True)

##########################################################################################
# Construct Labels
##########################################################################################

# note: this currently used the whole dataset to generate label_dict
print('Generating Label Dict', flush=True)
label_counts = generate_counts(labels, delimiter=',')
label_dict = write_dict_to_file(label_counts, output_prefix + 'labels.txt', args.min_class_count)
print('Labels File Written: ', output_prefix + 'labels.txt', flush=True)

##########################################################################################
# Generate Data
##########################################################################################

print('Writing Training Data', flush=True)
write_samples_to_file(train_sentences, train_labels, token_dict, label_dict, output_prefix + 'train_x.txt', output_prefix + 'train_y.txt')

print('Writing Validation Data', flush=True)
write_samples_to_file(val_sentences, val_labels, token_dict, label_dict, output_prefix + 'val_x.txt', output_prefix + 'val_y.txt')

print('Writing Testing Data', flush=True)
write_samples_to_file(test_sentences, test_labels, token_dict, label_dict, output_prefix + 'test_x.txt', output_prefix + 'test_y.txt')



