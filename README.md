# [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) - For Keras :zap: :smiley:

Implementation of the [**AdamW optimizer**](https://arxiv.org/abs/1711.05101)(**Ilya Loshchilov, Frank Hutter**) for Keras. 

## Usage
```
from AdamW import AdamW

adamw = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)
```
Then, nothing change compared to the usual usage of an optimizer in Keras after the definition of model's architecture
```
model = Sequential()
<definition of the model_architecture>
model.compile(loss="mse", optimizer=adamw, metrics=[metrics.mse], ...)
```

Note that the size of a batch (batch_size), number of training samples per epoch (samples_per_epoch) and the number of epochs (epochs) are necessary to the normalization of the weight decay ([paper](https://arxiv.org/abs/1711.05101), Section 4)

## Done 
- Weight decay added to the parameters optimization
- Normalized weight decay added 

## To be done (eventually - help is welcome)
- Cosine annealing
- Warm restarts

# Source

[**ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION**](https://arxiv.org/pdf/1412.6980v8.pdf), I. Loshchilov, F. Hutter

[**Fixing Weight Decay Regularization in Adam**](https://arxiv.org/pdf/1711.05101.pdf), D.P. Kingma, J. Lei Ba
