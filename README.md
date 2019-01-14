Keras extension packages
========================

This package provide tips of keras extentions.

#### Installation
1. check out this repository.
```
git clone https://github.com/k1414st/keras_extension
```
2. use setup.py to install this package.
```
python setup.py install
```

PartialConvND (Layer class, N = 1~3)
-----------------------
#### Description
PartialConvND classes are implementation of  "Partial Convolutional based padding" which re-weight convolution near image borders based on the ratios between the padded area and the convolution sliding window area.
This algorithm has been introduced by *Guilin Liu et al.* in https://arxiv.org/abs/1811.11718

#### usage
You can easily use PartialConv2D instead of Conv2D like this simple code.
```
# from keras.layers import Conv2D
from keras_extension.layers import PartialConv2D

# Conv2D(filters=3, kernel_size=(3, 3)),
PartialConv2D(filters=3, kernel_size=(3, 3)),
```


gelu (activation function)
-----------------------
#### Description

Gaussian Error Linear Unit (GELU), a high-performing neural network activation function. The GELU nonlinearity is the expected transformation of a stochastic regularizer which randomly applies the identity or zero map to a neuron's input. The GELU nonlinearity weights inputs by their magnitude, rather than gates inputs by their sign as in ReLUs. Gaussian Error Linear Unit (GELU), a high-performing neural network activation function.
This algorithm has been introduced by Dan Hendrycks et al. in https://arxiv.org/abs/1606.08415.

#### usage
```
from keras.layers import Dense, Activation
from keras_extension.activation import gelu
d = Dense(activation=gelu)
act = Activation(gelu)
```


MAC (recurrent layer of "Memory, Attention, and Composition" cell)
-----------------------
#### Description


MAC network, a novel fully differentiable neural network architecture, designed to facilitate explicit and expressive reasoning. MAC moves away from monolithic black-box neural architectures towards a design that encourages both transparency and versatility. The model approaches problems by decomposing them into a series of attention-based reasoning steps, each performed by a novel recurrent Memory, Attention, and Composition (MAC) cell that maintains a separation between control and memory. By stringing the cells together and imposing structural constraints that regulate their interaction, MAC effectively learns to perform iterative reasoning processes that are directly inferred from the data in an end-to-end approach.
This algorithm has been introduced by Drew A. Hudson et al. in https://arxiv.org/abs/1803.03067

#### usage
```
from keras.layers import Input
from keras_extension.layers import MAC
input_q = Input(shape=(1024,))
input_cw = Input(shape=(32, 512))
input_knowledge = Input(shape=(16*16, 512))

c, m = MAC(recurrent_length=16)([input_q, input_cw, input_knowledge])
```

GraphConv (generalized Graph Convolution Layer)
-----------------------
#### Description

Graph Convolution Layer operates over a graph representing both local and nonlocal dependencies between graphical relationed units (i.e. commercial distribution, SNS networks or relationed words or articles, etc). The algorithm propagates information between connected nodes through graph convolutions and exploits the richer representations to improve graphically relationed representations.
This algorithm has been introduced by Yujie-Qian et al. in https://arxiv.org/abs/1810.13083, and Liang Yao et al. in https://arxiv.org/abs/1809.05679.

#### usage
```
from keras.layers import Input
from keras_extension.layers import GraphConv
input_layer = Input(shape=(32, 256))
input_graph = Input(shape=(32, 32))

output_layer = GraphConv(units=16)([input_layer, input_graph])
```
