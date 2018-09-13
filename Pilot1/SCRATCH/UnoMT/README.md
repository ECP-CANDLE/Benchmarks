# UnoMT in Pytorch
Multi-tasking (drug response, cell line classification, etc.) Uno Implemented in PyTorch.
https://github.com/xduan7/UnoPytorch


## Todos
* More labels for the network like drug labels;
* Dataloader hanging problem when num_workers set to more than 0;
* Better pre-processing for drug descriptor integer features;
* Network regularization with weight decay and/or dropout;
* Hyper-parameter searching; 

## Prerequisites
```
Python          3.6.4
PyTorch         0.4.1
SciPy           1.1.0
pandas          0.23.4
Scikit-Learn    0.19.1
urllib3         1.23
joblib          0.12.2
```


The default network structure is shown below: 
<img src="https://github.com/xduan7/UnoPytorch/blob/master/images/default_network.jpg" width="100%">

An example of the program output for training on NCI60 and valdiation on all other data sources is shown below:
```
python3.6 ./launcher.py
Training Arguments:
{
    "trn_src": "NCI60",
    "val_srcs": [
        "NCI60",
        "CTRP",
        "GDSC",
        "CCLE",
        "gCSI"
    ],
    "grth_scaling": "none",
    "dscptr_scaling": "std",
    "rnaseq_scaling": "std",
    "dscptr_nan_threshold": 0.0,
    "qed_scaling": "none",
    "rnaseq_feature_usage": "source_scale",
    "drug_feature_usage": "both",
    "validation_ratio": 0.2,
    "disjoint_drugs": false,
    "disjoint_cells": true,
    "gene_layer_dim": 1024,
    "gene_latent_dim": 512,
    "gene_num_layers": 2,
    "drug_layer_dim": 4096,
    "drug_latent_dim": 2048,
    "drug_num_layers": 2,
    "autoencoder_init": true,
    "resp_layer_dim": 2048,
    "resp_num_layers_per_block": 2,
    "resp_num_blocks": 4,
    "resp_num_layers": 2,
    "resp_dropout": 0.0,
    "resp_activation": "none",
    "cl_clf_layer_dim": 256,
    "cl_clf_num_layers": 2,
    "drug_target_layer_dim": 512,
    "drug_target_num_layers": 2,
    "drug_qed_layer_dim": 512,
    "drug_qed_num_layers": 2,
    "drug_qed_activation": "sigmoid",
    "resp_loss_func": "mse",
    "resp_opt": "SGD",
    "resp_lr": 1e-05,
    "cl_clf_opt": "SGD",
    "cl_clf_lr": 0.01,
    "drug_target_opt": "SGD",
    "drug_target_lr": 0.01,
    "drug_qed_loss_func": "mse",
    "drug_qed_opt": "SGD",
    "drug_qed_lr": 0.01,
    "resp_val_start_epoch": 0,
    "early_stop_patience": 20,
    "lr_decay_factor": 0.98,
    "trn_batch_size": 32,
    "val_batch_size": 256,
    "max_num_batches": 1000,
    "max_num_epochs": 1000,
    "multi_gpu": false,
    "no_cuda": false,
    "rand_state": 0
}
RespNet(
  (_RespNet__gene_encoder): Sequential(
    (dense_0): Linear(in_features=942, out_features=1024, bias=True)
    (relu_0): ReLU()
    (dense_1): Linear(in_features=1024, out_features=1024, bias=True)
    (relu_1): ReLU()
    (dense_2): Linear(in_features=1024, out_features=512, bias=True)
  )
  (_RespNet__drug_encoder): Sequential(
    (dense_0): Linear(in_features=4688, out_features=4096, bias=True)
    (relu_0): ReLU()
    (dense_1): Linear(in_features=4096, out_features=4096, bias=True)
    (relu_1): ReLU()
    (dense_2): Linear(in_features=4096, out_features=2048, bias=True)
  )
  (_RespNet__resp_net): Sequential(
    (dense_0): Linear(in_features=2561, out_features=2048, bias=True)
    (activation_0): ReLU()
    (residual_block_0): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
      )
      (activation): ReLU()
    )
    (residual_block_1): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
      )
      (activation): ReLU()
    )
    (residual_block_2): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
      )
      (activation): ReLU()
    )
    (residual_block_3): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
      )
      (activation): ReLU()
    )
    (dense_1): Linear(in_features=2048, out_features=2048, bias=True)
    (res_relu_1): ReLU()
    (dense_2): Linear(in_features=2048, out_features=2048, bias=True)
    (res_relu_2): ReLU()
    (dense_out): Linear(in_features=2048, out_features=1, bias=True)
  )
)
================================================================================
Training Epoch   1:
	Drug Weighted QED Regression Loss: 0.055694
	Drug Response Regression Loss:  1871.18

Validation Results:
	Cell Line Classification: 
		Category Accuracy: 		98.98%; 
		Site Accuracy: 			80.95%; 
		Type Accuracy: 			82.76%
	Drug Target Family Classification Accuracy:  1.85%
	Drug Weighted QED Regression
		MSE: 0.028476 	 MAE: 0.137004 	 R2: +0.17
	Drug Response Regression:
		NCI60  	 MSE:  1482.07 	 MAE:    27.89 	 R2: +0.53
		CTRP   	 MSE:  2554.45 	 MAE:    38.62 	 R2: +0.27
		GDSC   	 MSE:  2955.78 	 MAE:    42.73 	 R2: +0.11
		CCLE   	 MSE:  2799.06 	 MAE:    42.44 	 R2: +0.31
		gCSI   	 MSE:  2601.50 	 MAE:    38.44 	 R2: +0.35
Epoch Running Time: 110.0 Seconds.
================================================================================
Training Epoch   2:
    ...
    ...

Program Running Time: 8349.6 Seconds.
================================================================================
Overall Validation Results:

	Best Results from Different Models (Epochs):
		Cell Line Categories     Best Accuracy: 99.474% (Epoch =   5)
		Cell Line Sites          Best Accuracy: 97.401% (Epoch =  60)
		Cell Line Types          Best Accuracy: 97.368% (Epoch =  40)
		Drug Target Family 	 Best Accuracy: 66.667% (Epoch =  23)
		Drug Weighted QED 	 Best R2 Score: +0.7422 (Epoch =  59, MSE = 0.008837, MAE = 0.069400)
		NCI60  	 Best R2 Score: +0.8107 (Epoch =  56, MSE =   601.18, MAE =  16.57)
		CTRP   	 Best R2 Score: +0.3945 (Epoch =  37, MSE =  2127.28, MAE =  31.44)
		GDSC   	 Best R2 Score: +0.2448 (Epoch =  22, MSE =  2506.03, MAE =  35.55)
		CCLE   	 Best R2 Score: +0.4729 (Epoch =   4, MSE =  2153.30, MAE =  33.63)
		gCSI   	 Best R2 Score: +0.4512 (Epoch =  31, MSE =  2203.04, MAE =  32.63)

	Best Results from the Same Model (Epoch =  22):
		Cell Line Categories     Accuracy: 99.408%
		Cell Line Sites          Accuracy: 97.138%
		Cell Line Types          Accuracy: 97.039%
		Drug Target Family 	 Accuracy: 57.407% 
		Drug Weighted QED 	 R2 Score: +0.6033 (MSE = 0.013601, MAE = 0.093341)
		NCI60  	 R2 Score: +0.7885 (MSE =   672.00, MAE =  17.89)
		CTRP   	 R2 Score: +0.3841 (MSE =  2163.66, MAE =  32.28)
		GDSC   	 R2 Score: +0.2448 (MSE =  2506.03, MAE =  35.55)
		CCLE   	 R2 Score: +0.4653 (MSE =  2184.62, MAE =  34.12)
		gCSI   	 R2 Score: +0.4271 (MSE =  2299.59, MAE =  32.93)
```

For default hyper parameters, the transfer learning matrix results are shown below:
<p align="center">
	<img src="https://github.com/xduan7/UnoPytorch/blob/master/images/default_results.jpg" width="80%">
</p>

Note that the green cells represents R2 score of higher than 0.1, red cells are R2 scores lower than -0.1 and yellows are for all the values in between. 
