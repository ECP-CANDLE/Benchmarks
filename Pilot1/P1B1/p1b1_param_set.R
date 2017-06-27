# see https://cran.r-project.org/web/packages/ParamHelpers/ParamHelpers.pdfmakeNum
# the parameter names should match names of the arguments expected by the benchmark

param.set <- makeParamSet(
  makeDiscreteParam("batch_size", values = c(32, 64, 128)),
  makeDiscreteParam("dense", values = c("1000", "1000 1000", "1000 1000 1000")),
  makeDiscreteParam("molecular_num_hidden", values = c("54 12", "32 16 8"),
  makeDiscreteParam("activation", values = c("relu", "sigmoid", "tanh")),
  makeDiscreteParam("optimizer", values = c("adam", "sgd", "rmsprop")),
  makeNumericParam("drop", lower = 0, upper = 0.6),
  makeIntegerParam("latent_dim", lower = 100, upper = 1000),
  makeIntegerParam("epochs", lower = 100, upper = 100),

  ## DEBUG PARAMETERS: DON'T USE THESE IN PRODUCTION RUN
  #  makeIntegerParam("feature_subsample", lower=500, upper=500),
  ## END DEBUG PARAMS
)
