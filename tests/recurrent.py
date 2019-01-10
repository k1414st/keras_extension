# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer


class ControlUnit(Layer):
    """Control Unit class for the MAC Cell.

    # Arguments
        attention_activation: Activation function to use
            for the attention step of contextual words.
            (see [activations](../activations.md)).
            Default: softmax.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 attention_activation='softmax',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ControlUnit, self).__init__(**kwargs)
        self.attention_activation = activations.get(attention_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        contextual_seq_shape = input_shapes[2]
        self.units = contextual_seq_shape[-1]

        def add_w(shape, name):
            return self.add_weight(shape=shape, name=name,
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.q_kernel = add_w((self.units * 2, self.units * 1), 'q_kernel')
        self.cq_kernel = add_w((self.units * 2, self.units * 1), 'cq_kernel')
        self.ca_kernel = add_w((self.units * 1, 1), 'ca_kernel')

        self.q_bias = add_w((self.units * 1,), 'q_bias')
        self.cq_bias = add_w((self.units * 1,), 'cq_bias')
        self.ca_bias = add_w((1,), 'ca_bias')
        self.built = True

    def call(self, inputs, training=None):

        c_prev, extractor, cw_s = inputs[0], inputs[1], inputs[2]

        q_i = K.bias_add(K.dot(extractor, self.q_kernel),
                         self.q_bias)
        cq_i = K.bias_add(K.dot(K.concatenate([c_prev, q_i], axis=-1),
                                self.cq_kernel),
                          self.cq_bias)
        cqcw = K.expand_dims(cq_i, axis=1) * cw_s
        ca_is = K.bias_add(K.dot(cqcw, self.ca_kernel),
                           self.ca_bias)
        cv_is = self.attention_activation(ca_is, axis=-1)
        c_i = K.sum(cv_is * cw_s, axis=1)
        return c_i

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.units)


class ReadUnit(Layer):
    """Read Unit class for the MAC Cell.

    # Arguments
        attention_activation: Activation function to use
            for the attention step of weighing knowledge.
            (see [activations](../activations.md)).
            Default: softmax.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 attention_activation='softmax',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ReadUnit, self).__init__(**kwargs)
        self.attention_activation = activations.get(attention_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        control_shape = input_shapes[0]
        self.units = control_shape[-1]

        def add_w(shape, name):
            return self.add_weight(shape=shape, name=name,
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.im_kernel = add_w((self.units*1, self.units*1), 'im_kernel')
        self.ik_kernel = add_w((self.units*1, self.units*1), 'im_kernel')
        self.id_kernel = add_w((self.units*2, self.units*1), 'im_kernel')
        self.ra_kernel = add_w((self.units*1, self.units*1), 'im_kernel')

        self.im_bias = add_w((self.units*1,), 'im_bias')
        self.ik_bias = add_w((self.units*1,), 'ik_bias')
        self.id_bias = add_w((self.units*1,), 'id_bias')
        self.ra_bias = add_w((self.units*1,), 'ra_bias')

        self.built = True

    def call(self, inputs, training=None):

        c_cur, m_prev, knowledge = inputs[0], inputs[1], inputs[2]

        Im = K.expand_dims(K.bias_add(K.dot(m_prev, self.im_kernel),
                                      self.im_bias), axis=1)
        Ik = K.bias_add(K.dot(knowledge, self.ik_kernel),
                        self.ik_bias)
        I = Im * Ik
        Id = K.bias_add(K.dot(K.concatenate([I, knowledge], axis=-1),
                              self.id_kernel),
                        self.id_bias)
        cI = K.expand_dims(c_cur, axis=1) * Id
        ra = K.bias_add(K.dot(cI, self.ra_kernel),
                        self.ra_bias)
        rv = self.attention_activation(ra, axis=1)
        r = K.sum(rv * knowledge, axis=1)
        return r

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.units)


class WriteUnit(Layer):
    """Write Unit class for the MAC Cell.

    # Arguments
        forget_activation: Forget Activation function
            used for the softmax step
            (see [activations](../activations.md)).
            Default: softmax.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 forget_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(WriteUnit, self).__init__(**kwargs)
        self.forget_activation = activations.get(forget_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        control_shape = input_shapes[0]
        self.units = control_shape[-1]

        def add_w(shape, name):
            return self.add_weight(shape=shape, name=name,
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.mi_kernel = add_w((self.units * 2, self.units * 1), 'mi_kernel')
        self.mdsa_kernel = add_w((self.units * 1, self.units * 1), 'mdsa_kernel')
        self.mdi_kernel = add_w((self.units * 1, self.units * 1), 'mdi_kernel')
        self.cd_kernel = add_w((self.units * 1, 1), 'cd_kernel')

        self.mi_bias = add_w((self.units * 1,), 'mi_bias')
        self.md_bias = add_w((self.units * 1,), 'md_bias')

        if self.unit_forget_bias:
            def forget_bias_initializer(_, *args, **kwargs):
                return initializers.Ones()((1,), *args, **kwargs)
        else:
            forget_bias_initializer = self.bias_initializer
        self.cd_bias = self.add_weight(shape=(1,),
                                       name='cd_bias',
                                       initializer=forget_bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
        self.built = True

    def call(self, inputs, training=None):

        c, r, m, msa = inputs[0], inputs[1], inputs[2], inputs[3]

        mi = K.bias_add(K.dot(K.concatenate([r, m], axis=-1),
                              self.mi_kernel),
                        self.mi_bias)

        md = K.bias_add(K.dot(msa, self.mdsa_kernel) +
                        K.dot(mi, self.mdi_kernel),
                        self.md_bias)
        cd = K.bias_add(K.dot(c, self.cd_kernel),
                        self.cd_bias)
        mi = self.forget_activation(cd) * m + \
            self.forget_activation(1-cd) * md
        return mi

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.units)


class MACCell(Layer):
    """MAC Cell class for the MAC Recurrent Layer.

    # Arguments
        attention_activation: Attention activation function
            used for the contextual words weight, knowledge weight, and
            memory self-attention weight.
            Default: softmax.
        forget_activation: Forget Activation function
            used for the output of memory in the WriteUnit.
            Default: sigmoid.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 attention_activation='softmax',
                 forget_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.control_unit = \
            ControlUnit(attention_activation=attention_activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint)
        self.read_unit = \
            ReadUnit(attention_activation=attention_activation,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)
        self.write_unit = \
            WriteUnit(forget_activation=forget_activation,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      unit_forget_bias=unit_forget_bias,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)
        super(MACCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # c_seq, m_seq, extractor, cw_s, knowledge
        nb = input_shapes[4][0]
        d = input_shapes[4][2]

        cmr_shape = (nb, d)  # control, memory, r
        ext_shape = input_shapes[2]
        cw_shape = input_shapes[3]
        k_shape = input_shapes[4]

        self.units = d

        self.control_unit.build([cmr_shape, ext_shape, cw_shape])
        self.read_unit.build([cmr_shape, cmr_shape, k_shape])
        self.write_unit.build([cmr_shape, cmr_shape, cmr_shape, cmr_shape])
        self.built = True

    def call(self, inputs, training=None):

        c, m, extractor, cw_s, knowledge, m_sa = \
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]

        c_next = self.control_unit.call([c, extractor, cw_s])
        r = self.read_unit.call([c_next, m, knowledge])
        m_next = self.write_unit.call([c_next, r, m, m_sa])

        return [c_next, m_next]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.units),
                (input_shape[0][0], self.units)]


class MAC(Layer):
    """MAC Recurrent Layer class for the MAC Network.

    # Arguments
        recurrent_length: Recurrent length of MAC cell
            used for the loop of calling MAC cell.
        attention_activation: Attention activation function
            used for the contextual words weight, knowledge weight, and
            memory self-attention weight.
            Default: softmax.
        forget_activation: Forget Activation function
            used for the output of memory in the WriteUnit.
            Default: sigmoid.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self,
                 recurrent_length,
                 attention_activation='softmax',
                 forget_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.recurrent_length = recurrent_length
        self.attention_activation = activations.get(attention_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.mac_cell = \
            MACCell(attention_activation=attention_activation,
                    forget_activation=forget_activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    unit_forget_bias=unit_forget_bias,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)
        super(MAC, self).__init__(**kwargs)

    def build(self, input_shapes):
        # extractor, cw, knowledge
        nb = input_shapes[2][0]
        d = input_shapes[2][2]

        cm_shape = (nb, d)
        ext_shape = input_shapes[0]
        cw_shape = input_shapes[1]
        k_shape = input_shapes[2]

        self.units = d

        # c_seq, m_seq, extractor, cw_s, knowledge
        self.mac_cell.build([cm_shape, cm_shape, ext_shape,
                             cw_shape, k_shape])

        def add_w(shape, name):
            return self.add_weight(shape=shape, name=name,
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.sv_kernel = add_w((self.units * 1, 1), 'sv_kernel')
        self.sv_bias = add_w((self.units * 1,), 'sv_bias')

        # finished
        self.built = True

    def call(self, inputs, training=None):

        extractor, cw_s, knowledge = \
            inputs[0], inputs[1], inputs[2]

        c_0 = K.zeros_like(cw_s[:, 0, :])
        m_0 = K.zeros_like(cw_s[:, 0, :])

        # first
        c, m = self.mac_cell.call(
            [c_0, m_0, extractor, cw_s, knowledge, m_0])

        c_seq = K.expand_dims(c, axis=1)
        m_seq = K.expand_dims(m, axis=1)

        # second ~
        for i in range(1, self.recurrent_length):
            # self-attention
            cc = c_seq * K.expand_dims(c, axis=1)
            sv = K.bias_add(K.dot(cc, self.sv_kernel),
                            self.sv_bias)
            sa = self.attention_activation(sv, axis=1)
            m_sa = K.sum(sa * m_seq, axis=1)  # self-attentioned m_1~i

            # MAC cell (main flow)
            c, m = self.mac_cell.call(
                [c, m, extractor, cw_s, knowledge, m_sa])

            # make list of control & memory (for next self-attention)
            c_seq = K.concatenate([c_seq, K.expand_dims(c, axis=1)], axis=1)
            m_seq = K.concatenate([m_seq, K.expand_dims(m, axis=1)], axis=1)

        return [c, m]

    def compute_output_shape(self, input_shape):
        return [(input_shape[1][0], self.units),
                (input_shape[1][0], self.units)]


from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers import LSTM


input_c = Input(shape=(10,))
input_q = Input(shape=(20,))
input_s = Input(shape=(7, 10))
x = ControlUnit()([input_c, input_q, input_s])
x = Dense(10)(x)
output_layer = Activation('softmax')(x)
model = Model([input_c, input_q, input_s], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())


input_c = Input(shape=(10,))
input_m = Input(shape=(10,))
input_k = Input(shape=(7, 10))
x = ReadUnit()([input_c, input_m, input_k])
x = Dense(10)(x)
output_layer = Activation('softmax')(x)
model = Model([input_c, input_m, input_k], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


input_c = Input(shape=(10,))
input_r = Input(shape=(10,))
input_m = Input(shape=(10,))
input_msa = Input(shape=(10,))
x = WriteUnit()([input_c, input_r, input_m, input_msa])
x = Dense(10)(x)
output_layer = Activation('softmax')(x)
model = Model([input_c, input_r, input_m, input_msa], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


from keras.layers import Add, Flatten
input_c = Input(shape=(10,))
input_m = Input(shape=(10,))
input_q = Input(shape=(20,))
input_cw = Input(shape=(15, 10,))
input_k = Input(shape=(35, 10,))
input_msa = Input(shape=(10,))
x, y = MACCell()([input_c, input_m, input_q, input_cw, input_k, input_msa])
z = Add()([x, y])
z = Dense(10)(z)
output_layer = Activation('softmax')(z)
print(x, y, z, output_layer)
model = Model([input_c, input_m, input_q, input_cw,
               input_k, input_msa], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


from keras.layers import Add, Flatten
input_q = Input(shape=(20,))
input_cw = Input(shape=(15, 10,))
input_k = Input(shape=(35, 10,))
x, y = MAC(recurrent_length=3)([input_q, input_cw, input_k])
z = Add()([x, y])
z = Dense(10)(z)
output_layer = Activation('softmax')(z)
model = Model([input_q, input_cw, input_k], output_layer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
