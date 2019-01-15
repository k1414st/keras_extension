import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras_extension.layers import GraphConv


input_layer = Input(shape=(32, 256))  # L, D
input_graph = Input(shape=(32, 32))  # L, L
g = GraphConv(10, normalize_graph_output=True)
x = g([input_layer, input_graph])

x = Flatten()(x)
output_layer = Dense(1, activation='tanh')(x)

mdl = Model([input_layer, input_graph], output_layer)
mdl.compile(optimizer='adam', loss='binary_crossentropy')
mdl.summary()


input_layer = Input(shape=(32, 256))  # L, D
input_graph = Input(shape=(32, 32))  # L, L
g1 = GraphConv(256, normalize_graph_output=True)
g2 = GraphConv(256, normalize_graph_output=True)
x = g1([input_layer, input_graph])
x = g1([x, input_graph])
print(input_layer.shape, x.shape)

x = Flatten()(x)
output_layer = Dense(1, activation='tanh')(x)

mdl = Model([input_layer, input_graph], output_layer)
mdl.compile(optimizer='adam', loss='binary_crossentropy')
mdl.summary()

