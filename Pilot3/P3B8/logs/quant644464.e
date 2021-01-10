
Lmod is automatically replacing "xl/16.1.1-5" with "gcc/6.4.0".


Due to MODULEPATH changes, the following have been reloaded:
  1) spectrum-mpi/10.3.1.2-20200121

Traceback (most recent call last):
  File "optimize.py", line 217, in <module>
    main()
  File "optimize.py", line 213, in main
    run(params)
  File "optimize.py", line 197, in run
    model, {torch.nn.Linear}, dtype=torch.qint8
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 285, in quantize_dynamic
    convert(model, mapping, inplace=True)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 365, in convert
    convert(mod, mapping, inplace=True)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 365, in convert
    convert(mod, mapping, inplace=True)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 365, in convert
    convert(mod, mapping, inplace=True)
  [Previous line repeated 4 more times]
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 366, in convert
    reassign[name] = swap_module(mod, mapping)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/quantization/quantize.py", line 395, in swap_module
    new_mod = mapping[type(mod)].from_float(mod)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/nn/quantized/dynamic/modules/linear.py", line 103, in from_float
    qlinear = Linear(mod.in_features, mod.out_features, dtype=dtype)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/nn/quantized/dynamic/modules/linear.py", line 35, in __init__
    super(Linear, self).__init__(in_features, out_features, bias_, dtype=dtype)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/nn/quantized/modules/linear.py", line 152, in __init__
    self._packed_params = LinearPackedParams(dtype)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/nn/quantized/modules/linear.py", line 20, in __init__
    self.set_weight_bias(wq, None)
  File "/gpfs/alpine/proj-shared/med106/envs/opence/lib/python3.6/site-packages/torch/nn/quantized/modules/linear.py", line 26, in set_weight_bias
    self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)
RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine
