# VERY Experimental
# Code is not optimized

import os
import numpy
import argparse
import matplotlib.pyplot as plt

from keras import optimizers, regularizers, layers, models, callbacks, losses
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import *
from keras.datasets import imdb
from keras.preprocessing import sequence

from capsulelayers import CapsuleLayer, PrimaryCap, Length
from write import fileWrite

words = 5000
length = 600

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=words)

X_train = sequence.pad_sequences(X_train, maxlen=length)
X_test = sequence.pad_sequences(X_test, maxlen=length)

#y_train = to_categorical(y_train.astype('float32'))
#y_test = to_categorical(y_test.astype('float32'))

#=================================

n_class = 1

loss_func = 'binary_crossentropy'

emb_output_dim = 150
emb_train = True

conv1_filt = 128
conv1_kern = 3
conv1_stride = 1
conv1_pad = 'valid'
conv1_act = 'relu'

caps1_dim = 8
caps1_n_channels = 32
caps1_kern = 3
caps1_stride = 2
caps1_pad = 'valid'

caps2_num = n_class # 1 neuron at last layer for binary classification
caps2_dim = 16

#=================================

embed_set = [emb_output_dim, emb_train]
conv1_set = [conv1_filt, conv1_kern, conv1_stride, conv1_pad, conv1_act]
caps1_set = [caps1_dim, caps1_n_channels, caps1_kern, caps1_stride, caps1_pad]
caps2_set = [caps2_num, caps2_dim]

#=================================


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)

    embed = layers.Embedding(output_dim=emb_output_dim, input_dim=length, trainable=emb_train)(x)

    conv1 = layers.Conv1D(filters=conv1_filt, kernel_size=conv1_kern, strides=conv1_stride, padding=conv1_pad, activation=conv1_act, name='conv1')(embed)

    primarycaps = PrimaryCap(conv1, dim_capsule=caps1_dim, n_channels=caps1_n_channels, kernel_size=caps1_kern, strides=caps1_stride, padding=caps1_pad)

    digitcaps = CapsuleLayer(num_capsule=caps2_num, dim_capsule=caps2_dim, routings=routings,
                             name='digitcaps')(primarycaps)

    out_caps = Length(name='capsnet')(digitcaps)

    train_model = models.Model(x,out_caps)
    return train_model


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def preds(y_true, y_pred):
	return y_true


def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    global loss_func

    if loss_func == 'margin_loss':
    	loss_func = margin_loss

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[loss_func],
                  loss_weights=[1.],
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
               validation_data=(x_test, y_test), callbacks=[log, tb, checkpoint, lr_decay])

    '''
    #======================================

    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]

	# Testing
    test = X_train[0:100]
    layer_outs = [func([test, 1.]) for func in functors]
    print(layer_outs[-1])

    #======================================
    '''

    print('Evaluating...')
    scores = model.evaluate(x_test, y_test, verbose=0)
    vloss, vacc = scores

    data = [args.epochs, words, length, n_class, embed_set, conv1_set, caps1_set, caps2_set, vloss, vacc]

    if type(loss_func) != str:
        loss_func = 'margin_loss'

    fileWrite(data, loss_func)

    print('Architecture appended to ./result/archs/archs.txt')

    if args.save:
    	model.save_weights(args.save_dir + '/trained_model.h5')
    	print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


model = CapsNet(input_shape=X_train.shape[1:],
                n_class=n_class, routings=3)

model.summary()

parser = argparse.ArgumentParser(description="Capsule Network on imdb.")
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.1, type=float,
                    help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('-r', '--routings', default=3, type=int,
                    help="Number of iterations used in routing algorithm. should > 0")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--save_dir', default='./result')
parser.add_argument('-t', '--testing', action='store_true',
                    help="Test the trained model on testing dataset")
parser.add_argument('-w', '--weights', default=None,
                    help="The path of the saved weights. Should be specified when testing")
parser.add_argument('--save', default=False,
					help="Save model. True or False")
args = parser.parse_args()
print(args)

if args.weights is not None:
    print('Weights loaded')
    model.load_weights(args.weights)
if not args.testing:
    print('Training')
    train(model=model, data=((X_train, y_train), (X_test, y_test)), args=args)
else:
    print('Testing')

    if args.weights is None:
        print('No weights are provided. Will test using random initialized weights.')

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  loss_weights=[1.],
                  metrics=['accuracy'])

    num = 0
    count = 10

    acc = model.evaluate(X_test[num:num+count], y_test[num:num+count], verbose=0)

    #========================
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]
    #========================

    print(y_test[num:num+count])

    #========================
	# Testing
    test = X_test[num:num+count]
    layer_outs = [func([test, 1.]) for func in functors]
    print(layer_outs[-1])
    #========================

    print('test acc:', acc[1])
