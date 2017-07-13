# Benchmarks
ECP-CANDLE Benchmarks


This repository contains the CANDLE Benchmark codes. These codes implement deep learning architectures that are relevant to problems in cancer. These architectures address problems at different biological scales, specifically problems at the molecular, cellular and population scales.

The naming conventions adopted reflect the different biological scales.

Pilot1 (P1) benchmarks are formed out of problems and data at the cellular level. The high level goal of the problem behind the P1 benchmarks is to predict drug response based on molecular features of tumor cells and drug descriptors.

Pilot2 (P2) benchmarks are formed out of problems and data at the molecular level. The high level goal of the problem behind the P2 benchmarks is molecular dynamic simulations of proteins involved in cancer, specifically the RAS protein.

Pilot3 (P3) benchmarks are formed out of problems and data at the population level. The high level goal of the problem behind the P3 benchmarks is to predict cancer recurrence in patients based on patient related data.

Each of the problems (P1,P2,P3) informed the implementation of specific benchmarks, so P1B3 would be benchmark 3 of problem 1. At this point, we will refer to a benchmark by it's problem area and benchmark number, so it's natural to talk of the P1B1 benchmark. In the course of development, two new benchmarks were added to Pilot 1, named NT3 and TC1, which do not currently fit the naming convention. This will be resolved in future. Inside each benchmark directory, there exists a readme file that contains an overview of the benchmark, a description of the data and expected outcomes along with instructions for running the benchmark code.


Over time, we will also be adding implementations that make use of different tensor frameworks. The primary (baseline) benchmarks are implemented using Keras, and are named with '_baseline' in the name, for example p3b1_baseline_keras2.py. 

Implementations that use alternative tensor frameworks, such as mxnet or neon, will have the name of the framework in the name. Examples can be seen in the P1B3 benchmark contribs/ directory, for example:
        p1b3_mxnet.py
        p1b3_neon.py


General guidelines
   
For the 0.0 Release, we only include those benchmarks which have been extensively tested with the Supervisor workflows. The initial release will therefore only include Pilot1/NT3, Pilot2/P2B1 and Pilot3/P3B1. As more benchmarks are tested with the latest Supervisor, these will be added into the release branch. 

One major difference between the Release 0.0 codes and the pre-release codes is the development of a uniform code structure and command-line interface. This is both to improve compatibility with the Supervisor interface (allowing experiments to sweep over hyperparameters in a consistent way) as well as to improve the readability of the codes for users. 

Each benchmark includes a default_model.txt which includes the basic information required to describe a model, as well as defining keywords for the Supervisor framework to sweep over. 

Structure of the default model file. 

The default model file is composed of a set of key-value pairs, required to define a network of the benchmark class in question, such as the number and size of the hidden layers, as well as the required data files and locations. Optional arguments are those for which it may be sufficient to use the (Keras) defaults, but should be specified to guarantee the desired performance. If they are not provided, the Keras defaults will be used. The keywords here are examples, and the actual names are defined in the benchmark-specific parser section. Thus, for example, a network could be designed with multiple sets of dense layers, each defined by keywords ‘dense1’, ‘dense2’ etc. Similarly, if you wished to implement various activation functions on different layers, these could be entered as ‘act1’, ‘act2’, ‘out_act’. 
The allowable values and interpretation of those values are defined in the respective common files and subroutines, but we generally attempt to provide translations of all the common Keras implementations. 

Keyword examples:

data_url	: base url and subdirectory path to the data needed to run the benchmark.
train_data	: file name for the training data
test_data 	: file name for the testing data (if needed)

model_name: meta data which is passed to the logger to allow creation of appropriate file paths

conv		: specification of convolutional layers. These are specified by a list of vectors. Each vector represents a single convolutional layer, ordered as [# of filters, filer size, stride] with the dimensionality of the layer being inferred from the vector length; vector length is 2N + 1 where N is the dimensionality of the filter. Thus, all filter and stride dimensions must be specified, even if the filter is a square or cube.  Thus, [[100, 10, 1],[100, 20, 1]] specifies two 1D convolutional layers, while [100, 10, 10, 1, 1] specifies a single 2D convolutional layer. 

dense		: specification of dense or fully-connected layers. These are specified as a list of integers, each representing the size of the output of the respective layer. 

activation 	: keyword describing the type of activation function applied to the output of a hidden layer. We provide mechanisms to implement the commonly used Keras activation functions (‘relu’, ‘sigmoid’,’tanh’)

loss		: keyword describing the loss function to be used. We provide mechanisms to implement the commonly used Keras loss functions (‘mse’, ‘binary_crossentropy’, ’categorical_crossentropy’, ’smoothL1’)

optimizer	: One of ‘sgd’, ‘rmsprop’, ‘adam’, ‘adagrad’, ‘adadelta’

initialization 	: One of ‘constant’, ’uniform’, ’normal’, ’glorot_uniform’, ’lecun_uniform’, ’he_normal’ 

metrics         : Metrics used to compute the accuracy when evaluating the trained model.

rng_seed        : Value for the random number seed. Otherwise a random seed is chosen. 

More examples can be found within the respective default model files, but are generally limited to numerical parameters which can substantially affect model performance, such as learning_rate, batch_size and so on. For non-numeric inputs, the available values will generally correspond to the appropriate Keras function. 

