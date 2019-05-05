""" 
Mittels diesen Code koennen Bilder aus einem Ordner gelesen werden und in das 
Tensorflow Datei Format formatiert werden. 
Dieser Code wurde fuer Python 2.7 geschrieben und getestet. Fuer andere Versionen muessten dementsprechende
Anpassungen durchgefuehrt werden

Dem Nutzer bleibt die Option Testdaten zufaellig aus den Trainingsbildern zu waehlen,
die Testbilder explizit anzugeben oder einen eigenen Pfad zu den Testbilder zu geben.
Dabei ist es wichtig, dass der Ordner des Trainingsbilder folgende Struktur aufweist:
./Trainingsbilder/
- /ersteKlasse/
- - Bild1.png
- - Bild2.png
- - ...
- /zweiteKlasse/
- - Bild1.png
- - Bild2.png
- - ...
- ...

WARNUNG: Ordner fuer Trainings- und Testbilder muessen soviele Unterordner wie Klassen haben.

Anschlie{\ss}end wird ein CNN definiert und anhand der Trainingsbilder trainiert. Das CNN
hat insgesamt drei Conolutional Layer und drei Pooling Layer

Dieser Code ist stark angelehnt von folgendem Projekt:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import tensorflow as tf
import os

BILDER_ROOT_PFAD = './bilder/train/' # Pfad zu den Trainingsbildern
BILDER_ROOT_PFAD_TEST = './bilder/test/' # Pfad zu den Trainingsbildern

# Image Parameters
N_CLASSES = 43 # Anzahl der Klassen
IMG_HEIGHT = 32 # Hoehe auf der alle Bilder skaliert werden
IMG_WIDTH = 32 # Breite auf der alle Bilder skaliert werden
CHANNELS = 3 # 3 fuer farbige Bilder, 1 fuer grautoene
MAX_TRAIN_SIZE = 300 # wieviele Bilder sollen maximal fuer das Training genommen werden (weniger ist moeglich)

# Trainings Parameter
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# CNN Parameter
dropout = 0.75 # Dropout, probability to keep units

CONV1_PARAM = [32,3] # [Anzahl Filter,Groe{\ss}e] fuer die erste Convolution
POOL1_PARAM = [2,2] # [Schrittweite,Groe{\ss}e] fuer das erste Pooling
CONV2_PARAM = [64,3] # [Anzahl Filter,Groe{\ss}e] fuer die zweite Convolution
POOL2_PARAM = [2,2] # [Anzahl Filter,Groe{\ss}e] fuer das zweite Pooling
CONV3_PARAM = [128,4] # [Anzahl Filter,Groe{\ss}e] fuer die dritte Convolution
POOL3_PARAM = [2,2] # [Anzahl Filter,Groe{\ss}e] fuer das dritte Pooling
FULLY_CON_OUT_N = 200 # Groe{\ss}e der Ausgabe des Fully Connected Layers

# Zum Speichern benoetigt

MODEL_NAME = 'gtsrbCNN'
SAVE_DIR = './out/'

BOOL_SAVE_CKPT = True
BOOL_SAVE_PBTXT = True
# -----------------------------------------------
# Funktion Definitionen
# -----------------------------------------------

# -----------------------------------------------
# Daten einlesen
# -----------------------------------------------

def read_images(dataset_path, batch_size):
  labels = list() # Platzhalter fuer die Label
  imagepaths = list() # Platzhalter fuer die Pfade
  klassen_ordner = [ordner for ordner in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, ordner))]
  print(klassen_ordner)
  if len(klassen_ordner)!=N_CLASSES:
    raise ValueError('Anzahl Unterordner ('+str(len(klassen_ordner))+') und Klassen ('+str(N_CLASSES)+') stimmt nicht ueberein.')
  label = 0
  klassen_ordner = [os.path.join(dataset_path, x) for x in klassen_ordner]
  for ordner in klassen_ordner:
    file_names = [os.path.join(ordner, f) for f in os.listdir(ordner) if f.endswith(".png")]
    if len(file_names) > MAX_TRAIN_SIZE:
       file_names = file_names[:MAX_TRAIN_SIZE]
    print('Klasse '+str(label)+' mit '+str(len(file_names))+' Bildern.')
    for f in file_names:
      imagepaths.append(f)
      labels.append(label)
    label+=1

  # Convert to Tensor
  imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  print('Erfolgreich Tensor erstellt.')
  # Build a TF Queue, shuffle data
  image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
  print('Erfolgreich Daten gemischt.')
  # Read images from disk
  image = tf.read_file(image)
  image = tf.image.decode_jpeg(image, channels=CHANNELS)
  print('Erfolgreich Bilder eingelesen.')

  # Resize images to a common size
  image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

  # Normalize
  image = image * 1.0/127.5 - 1.0
  print('Erfolgreich Bilder skaliert und normalisiert.')
  # Create batches
  X, Y = tf.train.batch([image, label], batch_size=batch_size,
                        capacity=batch_size * 8,
                        num_threads=4)

  return X, Y

# -----------------------------------------------
# Definition des CNN
# -----------------------------------------------
# Die Struktur des CNNs wurde angepasst. Als weitere Referenz wird auf
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# verwiesen. Dies ist der originelle Code, von dem auch die Annotationen stammen.

# Build the data input
X, Y = read_images(BILDER_ROOT_PFAD, batch_size)
X_test, Y_test = read_images(BILDER_ROOT_PFAD_TEST, batch_size)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
  # Define a scope for reusing the variables
  with tf.variable_scope('ConvNet', reuse=reuse):

    # Convolution Layer with 32 filters and a kernel size of 3
    conv1 = tf.layers.conv2d(x, CONV1_PARAM[0], CONV1_PARAM[1], activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, POOL1_PARAM[0], POOL1_PARAM[1])

    # Convolution Layer with 32 filters and a kernel size of 3
    conv2 = tf.layers.conv2d(conv1, CONV2_PARAM[0], CONV2_PARAM[1], activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv2 = tf.layers.max_pooling2d(conv2, POOL2_PARAM[0], POOL2_PARAM[1])

    # Convolution Layer with 32 filters and a kernel size of 3
    conv3 = tf.layers.conv2d(conv1, CONV3_PARAM[0], CONV3_PARAM[1], activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv3 = tf.layers.max_pooling2d(conv3, POOL3_PARAM[0], POOL3_PARAM[1])

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv3)

    # Fully connected layer (in contrib folder for now)
    fc1 = tf.layers.dense(fc1, FULLY_CON_OUT_N)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(fc1, n_classes)
    # Because 'softmax_cross_entropy_with_logits' already apply softmax,
    # we only apply softmax to testing network
    out = tf.nn.softmax(out) if not is_training else out

  return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X_test, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_test, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

  # Run the initializer
  sess.run(init)

  # Start the data queue
  tf.train.start_queue_runners()

  # Training cycle
  for step in range(1, num_steps+1):

    if step % display_step == 0:
      # Run optimization and calculate batch loss and accuracy
      _, loss, acc = sess.run([train_op, loss_op, accuracy])
      print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Testing Accuracy= " + \
            "{:.3f}".format(acc))
    else:
      # Only run the optimization op (backprop)
      sess.run(train_op)

  print("Optimization Finished!")
    
  if BOOL_SAVE_PBTXT:
    tf.train.write_graph(sess.graph, SAVE_DIR, MODEL_NAME+'.pbtxt')
    print('Model '+MODEL_NAME+' gespeichert in '+SAVE_DIR+' als PBTXT')
  if BOOL_SAVE_CKPT:
    # Save the variables to disk.
    save_path = saver.save(sess, SAVE_DIR+MODEL_NAME+'.ckpt')
    print('Model '+MODEL_NAME+' gespeichert in '+SAVE_DIR+' als CKPT')

