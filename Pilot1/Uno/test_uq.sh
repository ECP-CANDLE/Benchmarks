#!/bin/bash

# UQ model
python uno_holdoutUQ_data.py
if [ -f "save_default/infer_cell_ids" ]; then
  python uno_trainUQ_keras2.py --cp True --uq_exclude_cells_file 'save_default/infer_cell_ids' --save_weights save_default/saved.model.weights.h5
  python uno_inferUQ_keras2.py  --uq_infer_file save_default/infer_cell_ids --uq_infer_given_cells True \
	  --model_file save_default/'uno.A=relu.B=32.E=10.O=sgd.LOSS=qtl.LR=0.01.CF=r.DF=df.DR=0.1.L1000.D1=1000.D2=1000.D3=1000.model.h5' \
	  --weights_file save_default/saved.model.weights.h5 \
	  --n_pred 10
fi

