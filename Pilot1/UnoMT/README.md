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
python unoMT_baseline_pytorch.py --resp_val_start_epoch 2 --epochs 5
Importing candle utils for pytorch
Created unoMT benchmark
Configuration file: ./unoMT_default_model.txt
{'autoencoder_init': True,
 'cl_clf_layer_dim': 256,
 'cl_clf_lr': 0.008,
 'cl_clf_num_layers': 2,
 'cl_clf_opt': 'SGD',
 'disjoint_cells': True,
 'disjoint_drugs': False,
 'drop': 0.1,
 'drug_feature_usage': 'both',
 'drug_latent_dim': 1024,
 'drug_layer_dim': 4096,
 'drug_num_layers': 2,
 'drug_qed_activation': 'sigmoid',
 'drug_qed_layer_dim': 1024,
 'drug_qed_loss_func': 'mse',
 'drug_qed_lr': 0.01,
 'drug_qed_num_layers': 2,
 'drug_qed_opt': 'SGD',
 'drug_target_layer_dim': 1024,
 'drug_target_lr': 0.002,
 'drug_target_num_layers': 2,
 'drug_target_opt': 'SGD',
 'dscptr_nan_threshold': 0.0,
 'dscptr_scaling': 'std',
 'early_stop_patience': 5,
 'epochs': 1000,
 'gene_latent_dim': 512,
 'gene_layer_dim': 1024,
 'gene_num_layers': 2,
 'grth_scaling': 'none',
 'l2_regularization': 1e-05,
 'lr_decay_factor': 0.98,
 'max_num_batches': 1000,
 'qed_scaling': 'none',
 'resp_activation': 'none',
 'resp_layer_dim': 2048,
 'resp_loss_func': 'mse',
 'resp_lr': 1e-05,
 'resp_num_blocks': 4,
 'resp_num_layers': 2,
 'resp_num_layers_per_block': 2,
 'resp_opt': 'SGD',
 'resp_val_start_epoch': 0,
 'rnaseq_feature_usage': 'combat',
 'rnaseq_scaling': 'std',
 'rng_seed': 0,
 'save_path': 'save/unoMT',
 'solr_root': '',
 'timeout': 3600,
 'train_sources': 'NCI60',
 'trn_batch_size': 32,
 'val_batch_size': 256,
 'val_sources': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'],
 'val_split': 0.2}
Params:
{'autoencoder_init': True,
 'cl_clf_layer_dim': 256,
 'cl_clf_lr': 0.008,
 'cl_clf_num_layers': 2,
 'cl_clf_opt': 'SGD',
 'datatype': <class 'numpy.float32'>,
 'disjoint_cells': True,
 'disjoint_drugs': False,
 'drop': 0.1,
 'drug_feature_usage': 'both',
 'drug_latent_dim': 1024,
 'drug_layer_dim': 4096,
 'drug_num_layers': 2,
 'drug_qed_activation': 'sigmoid',
 'drug_qed_layer_dim': 1024,
 'drug_qed_loss_func': 'mse',
 'drug_qed_lr': 0.01,
 'drug_qed_num_layers': 2,
 'drug_qed_opt': 'SGD',
 'drug_target_layer_dim': 1024,
 'drug_target_lr': 0.002,
 'drug_target_num_layers': 2,
 'drug_target_opt': 'SGD',
 'dscptr_nan_threshold': 0.0,
 'dscptr_scaling': 'std',
 'early_stop_patience': 5,
 'epochs': 5,
 'experiment_id': 'EXP000',
 'gene_latent_dim': 512,
 'gene_layer_dim': 1024,
 'gene_num_layers': 2,
 'gpus': [],
 'grth_scaling': 'none',
 'l2_regularization': 1e-05,
 'logfile': None,
 'lr_decay_factor': 0.98,
 'max_num_batches': 1000,
 'multi_gpu': False,
 'no_cuda': False,
 'output_dir': '/home/jamal/Code/ECP/CANDLE/Benchmarks/Pilot1/UnoMT/Output/EXP000/RUN000',
 'qed_scaling': 'none',
 'resp_activation': 'none',
 'resp_layer_dim': 2048,
 'resp_loss_func': 'mse',
 'resp_lr': 1e-05,
 'resp_num_blocks': 4,
 'resp_num_layers': 2,
 'resp_num_layers_per_block': 2,
 'resp_opt': 'SGD',
 'resp_val_start_epoch': 2,
 'rnaseq_feature_usage': 'combat',
 'rnaseq_scaling': 'std',
 'rng_seed': 0,
 'run_id': 'RUN000',
 'save_path': 'save/unoMT',
 'shuffle': False,
 'solr_root': '',
 'timeout': 3600,
 'train_bool': True,
 'train_sources': 'NCI60',
 'trn_batch_size': 32,
 'val_batch_size': 256,
 'val_sources': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'],
 'val_split': 0.2,
 'verbose': None}
Parameters initialized
Failed to split NCI60 cells in stratified way. Splitting randomly ...
Failed to split NCI60 cells in stratified way. Splitting randomly ...
Failed to split CCLE cells in stratified way. Splitting randomly ...
Failed to split CCLE drugs stratified on growth and correlation. Splitting solely on avg growth ...
Failed to split gCSI drugs stratified on growth and correlation. Splitting solely on avg growth ...
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
    (dense_2): Linear(in_features=4096, out_features=1024, bias=True)
  )
  (_RespNet__resp_net): Sequential(
    (dense_0): Linear(in_features=1537, out_features=2048, bias=True)
    (activation_0): ReLU()
    (residual_block_0): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_0): Dropout(p=0.1)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_1): Dropout(p=0.1)
      )
      (activation): ReLU()
    )
    (residual_block_1): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_0): Dropout(p=0.1)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_1): Dropout(p=0.1)
      )
      (activation): ReLU()
    )
    (residual_block_2): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_0): Dropout(p=0.1)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_1): Dropout(p=0.1)
      )
      (activation): ReLU()
    )
    (residual_block_3): ResBlock(
      (block): Sequential(
        (res_dense_0): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_0): Dropout(p=0.1)
        (res_relu_0): ReLU()
        (res_dense_1): Linear(in_features=2048, out_features=2048, bias=True)
        (res_dropout_1): Dropout(p=0.1)
      )
      (activation): ReLU()
    )
    (dense_1): Linear(in_features=2048, out_features=2048, bias=True)
    (dropout_1): Dropout(p=0.1)
    (res_relu_1): ReLU()
    (dense_2): Linear(in_features=2048, out_features=2048, bias=True)
    (dropout_2): Dropout(p=0.1)
    (res_relu_2): ReLU()
    (dense_out): Linear(in_features=2048, out_features=1, bias=True)
  )
)
Data sizes:
Train:
Data set: NCI60 Size: 882873
Validation:
Data set: NCI60 Size: 260286
Data set: CTRP Size: 1040021
Data set: GDSC Size: 235812
Data set: CCLE Size: 17510
Data set: gCSI Size: 10323
================================================================================
Training Epoch   1:
	Drug Weighted QED Regression Loss: 0.022274
	Drug Response Regression Loss:  1881.89
Epoch Running Time: 13.2 Seconds.
================================================================================
Training Epoch   2:
	Drug Weighted QED Regression Loss: 0.019416
	Drug Response Regression Loss:  1348.13
Epoch Running Time: 12.9 Seconds.
================================================================================
Training Epoch   3:
	Drug Weighted QED Regression Loss: 0.015868
	Drug Response Regression Loss:  1123.27
	Cell Line Classification: 
		Category Accuracy: 		99.01%; 
		Site Accuracy: 			94.11%; 
		Type Accuracy: 			94.18%
	Drug Target Family Classification Accuracy: 44.44%
	Drug Weighted QED Regression
		MSE: 0.018845 	 MAE: 0.111807 	 R2: +0.45
	Drug Response Regression:
		NCI60  	 MSE:   973.04 	 MAE:    22.18 	 R2: +0.69
		CTRP   	 MSE:  2404.64 	 MAE:    34.04 	 R2: +0.32
		GDSC   	 MSE:  2717.81 	 MAE:    36.53 	 R2: +0.19
		CCLE   	 MSE:  2518.47 	 MAE:    36.60 	 R2: +0.38
		gCSI   	 MSE:  2752.33 	 MAE:    36.97 	 R2: +0.35
Epoch Running Time: 54.6 Seconds.
================================================================================
Training Epoch   4:
	Drug Weighted QED Regression Loss: 0.014096
	Drug Response Regression Loss:   933.27
	Cell Line Classification: 
		Category Accuracy: 		99.34%; 
		Site Accuracy: 			96.12%; 
		Type Accuracy: 			96.18%
	Drug Target Family Classification Accuracy: 44.44%
	Drug Weighted QED Regression
		MSE: 0.018467 	 MAE: 0.110287 	 R2: +0.46
	Drug Response Regression:
		NCI60  	 MSE:   844.51 	 MAE:    20.41 	 R2: +0.73
		CTRP   	 MSE:  2314.19 	 MAE:    33.76 	 R2: +0.35
		GDSC   	 MSE:  2747.73 	 MAE:    36.65 	 R2: +0.18
		CCLE   	 MSE:  2482.03 	 MAE:    35.89 	 R2: +0.39
		gCSI   	 MSE:  2665.35 	 MAE:    36.27 	 R2: +0.37
Epoch Running Time: 54.9 Seconds.
================================================================================
Training Epoch   5:
	Drug Weighted QED Regression Loss: 0.013514
	Drug Response Regression Loss:   846.06
	Cell Line Classification: 
		Category Accuracy: 		99.38%; 
		Site Accuracy: 			95.89%; 
		Type Accuracy: 			95.30%
	Drug Target Family Classification Accuracy: 44.44%
	Drug Weighted QED Regression
		MSE: 0.017026 	 MAE: 0.106697 	 R2: +0.50
	Drug Response Regression:
		NCI60  	 MSE:   835.82 	 MAE:    21.33 	 R2: +0.74
		CTRP   	 MSE:  2653.04 	 MAE:    37.98 	 R2: +0.25
		GDSC   	 MSE:  2892.86 	 MAE:    39.76 	 R2: +0.13
		CCLE   	 MSE:  2412.75 	 MAE:    36.82 	 R2: +0.41
		gCSI   	 MSE:  2888.99 	 MAE:    38.70 	 R2: +0.32
Epoch Running Time: 55.5 Seconds.
Program Running Time: 191.1 Seconds.
================================================================================
Overall Validation Results:

	Best Results from Different Models (Epochs):
		Cell Line Categories     Best Accuracy: 99.375% (Epoch =   5)
		Cell Line Sites          Best Accuracy: 96.118% (Epoch =   4)
		Cell Line Types          Best Accuracy: 96.184% (Epoch =   4)
		Drug Target Family 	 Best Accuracy: 44.444% (Epoch =   3)
		Drug Weighted QED 	 Best R2 Score: +0.5034 (Epoch =   5, MSE = 0.017026, MAE = 0.106697)
		NCI60  	 Best R2 Score: +0.7369 (Epoch =   5, MSE =   835.82, MAE =  21.33)
		CTRP   	 Best R2 Score: +0.3469 (Epoch =   4, MSE =  2314.19, MAE =  33.76)
		GDSC   	 Best R2 Score: +0.1852 (Epoch =   3, MSE =  2717.81, MAE =  36.53)
		CCLE   	 Best R2 Score: +0.4094 (Epoch =   5, MSE =  2412.75, MAE =  36.82)
		gCSI   	 Best R2 Score: +0.3693 (Epoch =   4, MSE =  2665.35, MAE =  36.27)

	Best Results from the Same Model (Epoch =   5):
		Cell Line Categories     Accuracy: 99.375%
		Cell Line Sites          Accuracy: 95.888%
		Cell Line Types          Accuracy: 95.296%
		Drug Target Family 	 Accuracy: 44.444% 
		Drug Weighted QED 	 R2 Score: +0.5034 (MSE = 0.017026, MAE = 0.106697)
		NCI60  	 R2 Score: +0.7369 (MSE =   835.82, MAE =  21.33)
		CTRP   	 R2 Score: +0.2513 (MSE =  2653.04, MAE =  37.98)
		GDSC   	 R2 Score: +0.1327 (MSE =  2892.86, MAE =  39.76)
		CCLE   	 R2 Score: +0.4094 (MSE =  2412.75, MAE =  36.82)
		gCSI   	 R2 Score: +0.3164 (MSE =  2888.99, MAE =  38.70)
```

For default hyper parameters, the transfer learning matrix results are shown below:
<p align="center">
	<img src="https://github.com/xduan7/UnoPytorch/blob/master/images/default_results.jpg" width="80%">
</p>

Note that the green cells represents R2 score of higher than 0.1, red cells are R2 scores lower than -0.1 and yellows are for all the values in between. 
