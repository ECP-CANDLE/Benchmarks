# see
  https://cran.r-project.org/web/packages/ParamHelpers/ParamHelpers.pdfmakeNum
# the parameter names should match names of the arguments expected by
# the benchmark
param.set <- makeParamSet(
  makeDiscreteParam("learning_rate", values = c(0.01, 0.005, 0.015, 0.02)),
  makeDiscreteParam("batch_size", values = c(32, 64, 128)),
  makeDiscreteParam("molecular_num_hidden", values = c("256 128 64 32 16 8", "256 128 64 32 16", "512 256 128 64", "512 256 128 64 32"),
  makeDiscreteParam("sampling_density", values = c(0.1, 0.25, 0.5, 1.0)

  ## DEBUG PARAMETERS: DON'T USE THESE IN PRODUCTION RUN
#  makeIntegerParam("feature_subsample", lower=500, upper=500),
#  makeIntegerParam("train_steps", lower=100, upper=100),
#  makeIntegerParam("val_steps", lower=10, upper=10),
#  makeIntegerParam("test_steps", lower=10, upper=10),
#  makeIntegerParam("epochs", lower = 3, upper = 3)
  ## END DEBUG PARAMS
)

#  makeDiscreteParam("num_hidden", values = c("512 256 128 64 32 16", "512 64")),
#  makeDiscreteParam("dense", values = c("500 100 50", "1000 500 100 50", "2000 1000 500 100 50")),
#  makeDiscreteParam("activation", values = c("relu", "sigmoid", "tanh")),
#  makeDiscreteParam("optimizer", values = c("adam", "sgd", "rmsprop")),
#  makeNumericParam("drop", lower = 0, upper = 0.5),
#  makeDiscreteParam("conv", values = c("0 0 0", "5 5 1", "10 10 1 5 5 1")),
