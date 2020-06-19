
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6,7' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import pandas as pd
import numpy as np
import imageio
import PIL.Image
from sklearn.metrics import classification_report
import scipy.misc 
from scipy.ndimage.interpolation import zoom, rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.stats.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import pylab as P
from tensorflow.python.platform import gfile
from skimage.transform import resize
from tensorflow.core.framework import variable_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.framework.ops import register_proto_function
from argparse import ArgumentParser
from collections import defaultdict
from scipy.stats import ttest_ind,wilcoxon,ranksums
from keras.applications.inception_v3 import InceptionV3

import tensorflow.contrib.graph_editor as ge

random.seed(1)
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import saliency

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from skimage import exposure 
from keras.layers import Dropout, Flatten, Dense, Input, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import Model
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras import backend as K 
from keras import optimizers, layers, utils
from keras.applications import DenseNet121

tf.logging.set_verbosity(tf.logging.ERROR)

print('GPU TEST', tf.test.is_gpu_available())


parser = ArgumentParser(description = "choose randomized graph")
parser.add_argument('--filename', '-f', default = 'inception_2_0.ckpt')
parser.add_argument('--start', '-s', default = '0')
parser.add_argument('--end', '-e', default = '1067')
args = parser.parse_args()

def get_model(base_model, 
              layer, 
              lr=1e-4, 
              input_shape=(224,224,3), 
              classes=2,
              activation="softmax",
              dropout=None,  
              pooling="avg", 
              weights=None,
              pretrained="imagenet"): 
    base = base_model(input_shape=input_shape,
                      include_top=False,
                      weights=pretrained, 
                      ) 
    if pooling == "avg": 
        x = GlobalAveragePooling2D()(base.output) 
    elif pooling == "max": 
        x = GlobalMaxPooling2D()(base.output) 
    elif pooling is None: 
        x = Flatten()(base.output) 
    if dropout is not None: 
        x = Dropout(dropout)(x) 
    x = Dense(classes, activation=activation)(x) 
    model = Model(inputs=base.input, outputs=x) 
    if weights is not None: 
        model.load_weights(weights) 
    # for l in model.layers[:layer]:
    #     l.trainable = False 
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], 
                  optimizer=optimizers.Adam(lr)) 
    return model

def preprocess_input(x, model):
    x = x.astype("float32")
    if model in ("inception","xception","mobilenet"): 
        x /= 255.
        x -= 0.5
        x *= 2.
    if model in ("densenet"): 
        # x /= 255.
       if x.shape[-1] == 3:
           x[..., 0] -= 0.485
           x[..., 1] -= 0.456
           x[..., 2] -= 0.406 
           x[..., 0] /= 0.229 
           x[..., 1] /= 0.224
           x[..., 2] /= 0.225 
       elif x.shape[-1] == 1: 
           x[..., 0] -= 0.449
           x[..., 0] /= 0.226
    elif model in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

def soft_dice(s_map, bb_coords):
    #extracts saliency maps
    s_map = np.squeeze(s_map)
    bb_mask = np.zeros(s_map.shape)
    bb_mask2 = np.ones(s_map.shape)
    for bbox in bb_coords:
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        bb_mask[y_min:(y_max+1),x_min:(x_max+1)] = 1
        bb_mask2[y_min:(y_max+1),x_min:(x_max+1)] = 0
      #bb_mask[(319-y_max):(319-(y_min+1)),x_min:(x_max+1)] = 1
    
    intersect = 2*np.sum(np.multiply(s_map,bb_mask))
    intersect2 = 2*np.sum(np.multiply(s_map,bb_mask2))

    union = np.sum(s_map) + np.sum(bb_mask)
    union2 = np.sum(s_map) + np.sum(bb_mask2)
    
    iou_soft = intersect/union
    iou_soft2 = intersect2/union2

    return iou_soft,iou_soft2

def saliency_ttest(s_map,bb_coords2,prediction_class):

  #extracts saliency maps
  s_map = np.squeeze(s_map)
  tmp = s_map
  s_map = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
  bb_mask = bb_coords2

  s_map_bb = s_map * bb_mask
  s_map_bb_null = s_map * (1-bb_mask)

  # soft_dice_in, soft_dice_out = soft_dice(s_map,bb_coords1)
  hist_in = np.histogram(s_map_bb,100,range = (0,1.))
  hist_out = np.histogram(s_map_bb_null,100,range = (0,1.))

  s_map_bb_mean = np.mean(s_map_bb)
  s_map_bb_std = np.std(s_map_bb)
  s_map_bb_count = len(s_map_bb)
  s_map_bb_max = np.max(s_map_bb)
  s_map_bb_min = np.min(s_map_bb)

  #s_map_bb_summary_stats = [s_map_bb_mean,s_map_bb_std,s_map_bb_count,s_map_bb_max,s_map_bb_min]

  s_map_bb_null_mean = np.mean(s_map_bb_null)
  s_map_bb_null_std = np.std(s_map_bb_null)
  s_map_bb_null_count = len(s_map_bb_null)
  s_map_bb_null_max = np.max(s_map_bb_null)
  s_map_bb_null_min = np.min(s_map_bb_null)

  #s_map_bb_null_summary_stats = [s_map_bb_null_mean,s_map_bb_null_std,s_map_bb_null_count,s_map_bb_null_max,s_map_bb_null_min]

  t_eq, p_eq = ttest_ind(s_map_bb,s_map_bb_null)
  t_uneq, p_uneq = ttest_ind(s_map_bb,s_map_bb_null,equal_var=False)

  #converting to one-sided t-test values (comment out if you want a two-sided test)
  p_eq = p_eq/2
  p_uneq = p_uneq/2

  test_summary = [bb_coords2,s_map,prediction_class,0.,0.,0.,hist_in,hist_out] #change as necessary to return required stats

  return test_summary


def LoadImage(file_path):
  im = imageio.imread(file_path)
  im =  preprocess_input(im,"inception")
  im = np.expand_dims(im,axis=-1)
  return np.concatenate((im,im,im), axis = 2)

def LoadImage2(file_path):
  return imageio.imread(file_path)

root_dir = '/data/2015P002510/Bryan/Pneumothorax_Data/PNGs/'

sess = tf.Session()
input_size = 320
model = get_model(InceptionV3, 0, 7e-5, dropout=None, input_shape=(input_size,input_size,3))

saver = tf.train.Saver()
saver.restore(sess,'../graphs/' + args.filename)

with sess.graph.as_default():
  inp = sess.graph.get_tensor_by_name('input_1:0')
  logits = sess.graph.get_tensor_by_name('dense_1/BiasAdd:0')
  prediction = tf.argmax(logits,1)
  neuron_selector = tf.placeholder(tf.int32)
  y = logits[0][neuron_selector]


train_labels = '/data/2015P002510/Bryan/Pneumothorax_Data/csvs/train-rle.csv' #main SIIM Pneumothorax csv

df = pd.read_csv(train_labels)
print(df.columns)
df = df.drop_duplicates(subset=['ImageId'])

x_train = df[['ImageId']]
y_train = df[[' EncodedPixels']]

y_train = y_train.values
x_train = x_train.values

for i,yi in enumerate(y_train):
  try:
    if int(yi[0]) == -1:
      y_train[i] = 0
    else:
      y_train[i] = 1
  except:
    y_train[i] = 1


y_train = np.squeeze(np.array(y_train).astype(np.uint8))
# y_train = np.eye(2)[y_train]
x_train = np.squeeze(np.array(x_train))
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.1,random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=1)

scores = defaultdict(list)

print('Number of test images is ', len(x_test))

mask = np.zeros((len(x_test)))
indices = []
cnt = 0
for i in range(len(x_test)):
  if y_test[i] == 1:
    indices.append(i)
    cnt += 1

random.shuffle(indices)

for i in range(100):
  mask[indices[i]] = 1


if args.filename[-6] == 'n':
  ix = 0
elif args.filename[-7] == '_':
  ix = int(args.filename[-6])
else:
  ix = 10 + int(args.filename[-6])

cnt = 0

for j,imgid in enumerate(x_test[int(args.start):int(args.end)]):
  i = j+int(args.start)

  if y_test[i] == 1:
    img = LoadImage(root_dir + 'train/' + imgid + '.png')
    bb_coords = LoadImage2(root_dir + 'mask/' + imgid + '.png')
    l = sess.run(logits, feed_dict = {inp: [img]})[0]
    prediction_class = sess.run(prediction, feed_dict = {inp: [img]})[0]

    gbp = saliency.GuidedBackprop(sess.graph, sess, y, inp)
    gbp_mask = gbp.GetMask(img, feed_dict = {neuron_selector: prediction_class})
    gbp_mask = saliency.VisualizeImageGrayscale(gbp_mask)
    scores['gbp'].append(saliency_ttest(gbp_mask,bb_coords,prediction_class))

    gradient_saliency = saliency.GradientSaliency(sess.graph, sess, y, inp)
    vanilla_mask_3d = gradient_saliency.GetMask(np.reshape(img,(320,320,3)), feed_dict = {neuron_selector: prediction_class})
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(np.reshape(img,(320,320,3)), feed_dict = {neuron_selector: prediction_class})
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)    
    scores['grad'].append(saliency_ttest(vanilla_mask_grayscale,bb_coords,prediction_class))
    scores['sg'].append(saliency_ttest(smoothgrad_mask_grayscale,bb_coords,prediction_class))

    integrated_gradients = saliency.IntegratedGradients(sess.graph, sess, y, inp)
    baseline = np.zeros(img.shape)
    baseline.fill(-1)
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
      img, feed_dict = {neuron_selector: prediction_class}, x_steps=10, x_baseline=baseline)
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
      img, feed_dict = {neuron_selector: prediction_class}, x_steps=10, x_baseline=baseline)
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    scores['ig'].append(saliency_ttest(vanilla_mask_grayscale,bb_coords,prediction_class))
    scores['sig'].append(saliency_ttest(smoothgrad_mask_grayscale,bb_coords,prediction_class))

    gradCam = saliency.GradCam(sess.graph, sess, y, inp, conv_layer = sess.graph.get_tensor_by_name('mixed10/concat:0'))
    grad_mask_2d = gradCam.GetMask(img, feed_dict = {neuron_selector: prediction_class}, 
                                        should_resize = True, 
                                        three_dims = True)
    grad_mask_2d = saliency.VisualizeImageGrayscale(grad_mask_2d)
    scores['gcam'].append(saliency_ttest(grad_mask_2d,bb_coords,prediction_class))

    ggcam = gbp_mask * grad_mask_2d
    scores['ggcam'].append(saliency_ttest(ggcam,bb_coords,prediction_class))

    xrai = saliency.XRAI(sess.graph, sess, y, inp)
    xrai_mask = xrai.GetMask(img, feed_dict = {neuron_selector: prediction_class})
    xrai_mask = saliency.VisualizeImageGrayscale(xrai_mask)
    scores['xrai'].append(saliency_ttest(xrai_mask,bb_coords,prediction_class))


temp = scores['grad']
if os.path.isfile('../../scores4/gradient_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/gradient_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['grad'])))
np.save('../../scores4/gradient_scores_{}.npy'.format(ix),temp)

temp = scores['sg']
if os.path.isfile('../../scores4/smoothgradient_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/smoothgradient_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['sg'])))
np.save('../../scores4/smoothgradient_scores_{}.npy'.format(ix),temp)

temp = scores['ig']
if os.path.isfile('../../scores4/ig_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/ig_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['ig'])))
np.save('../../scores4/ig_scores_{}.npy'.format(ix),temp)

temp = scores['sig']
if os.path.isfile('../../scores4/smoothig_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/smoothig_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['sig'])))
np.save('../../scores4/smoothig_scores_{}.npy'.format(ix),temp)

temp = scores['gcam']
if os.path.isfile('../../scores4/gradcam_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/gradcam_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['gcam'])))
np.save('../../scores4/gradcam_scores_{}.npy'.format(ix),temp)

temp = scores['xrai']
if os.path.isfile('../../scores4/xrai_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/xrai_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['xrai'])))
np.save('../../scores4/xrai_scores_{}.npy'.format(ix),temp)

temp = scores['gbp']
if os.path.isfile('../../scores4/gbp_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/gbp_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['gbp'])))
np.save('../../scores4/gbp_scores_{}.npy'.format(ix),temp)

temp = scores['ggcam']
if os.path.isfile('../../scores4/ggcam_scores_{}.npy'.format(ix)):
  temp = np.load('../../scores4/ggcam_scores_{}.npy'.format(ix))
  temp = np.concatenate((temp,np.array(scores['ggcam'])))
np.save('../../scores4/ggcam_scores_{}.npy'.format(ix),temp)



