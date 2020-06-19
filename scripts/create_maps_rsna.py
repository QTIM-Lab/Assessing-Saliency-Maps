
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import imageio
from collections import defaultdict
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
#from scipy.special import softmax
from tensorflow.core.framework import variable_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.framework.ops import register_proto_function
from argparse import ArgumentParser
from scipy.stats import ttest_ind,wilcoxon,ranksums
from keras.applications.inception_v3 import InceptionV3

import tensorflow.contrib.graph_editor as ge

random.seed(1)
print('test')
import tensorflow as tf
import saliency

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from skimage import exposure 
from keras.layers import Dropout, Flatten, Dense, Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import Model
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras import backend as K 
from keras import optimizers, layers, utils
from keras.applications import DenseNet121

print('GPU TEST', tf.test.is_gpu_available())


parser = ArgumentParser(description = "choose randomized graph")
parser.add_argument('--filename', '-f', default = 'inception_1_0.ckpt')
parser.add_argument('--start', '-s', default = '0')
parser.add_argument('--end', '-e', default = '1472')
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
        x /= 255.
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

def saliency_ttest(s_map,bb_coords,prediction_class):

  #extracts saliency maps
  s_map = np.squeeze(s_map)
  tmp = s_map
  s_map = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
  bb_mask = np.zeros(s_map.shape)
  area = 0
  for bbox in bb_coords:
    x_min = min(int(bbox[0]),int(bbox[2]))
    y_min = min(int(bbox[1]),int(bbox[3]))
    x_max = max(int(bbox[0]),int(bbox[2]))
    y_max = max(int(bbox[1]),int(bbox[3]))
    area += (y_max - y_min)*(x_max - x_min)

    bb_mask[y_min:(y_max+1),x_min:(x_max+1)] = 1

  s_map_bb = s_map[bb_mask==1]
  s_map_bb_null = s_map[bb_mask==0]

  soft_dice_in, soft_dice_out = soft_dice(s_map,bb_coords)
  hist_in = np.histogram(s_map_bb,100,range = (0,1.))
  hist_out = np.histogram(s_map_bb_null,100,range = (0,1.))

  s_map_bb_mean = np.mean(s_map_bb)
  s_map_bb_std = np.std(s_map_bb)
  s_map_bb_count = len(s_map_bb)
  s_map_bb_max = np.max(s_map_bb)
  s_map_bb_min = np.min(s_map_bb)

  s_map_bb_null_mean = np.mean(s_map_bb_null)
  s_map_bb_null_std = np.std(s_map_bb_null)
  s_map_bb_null_count = len(s_map_bb_null)
  s_map_bb_null_max = np.max(s_map_bb_null)
  s_map_bb_null_min = np.min(s_map_bb_null)

  t_eq, p_eq = ttest_ind(s_map_bb,s_map_bb_null)
  t_uneq, p_uneq = ttest_ind(s_map_bb,s_map_bb_null,equal_var=False)

  #converting to one-sided t-test values (comment out if you want a two-sided test)
  p_eq = p_eq/2
  p_uneq = p_uneq/2

  test_summary = [bb_coords,s_map,prediction_class,area,soft_dice_in,soft_dice_out,hist_in,hist_out] #add or remove stats as required

  return test_summary


def LoadImage(file_path):
	im = imageio.imread(file_path)
	im = np.expand_dims(im,axis=-1)
	im = np.concatenate((im,im,im), axis = 2)
	return preprocess_input(im,"inception")

def downsamp_coord2(coord): #downsample bounding box coordinates
  coord_new = coord/1023 #it is actually (coord-0)/1023-0, but I simplified for computation
  coord_new = int(320*coord_new)
  return coord_new

root_dir = '/data/2015P002510/praveer/WeaklySupervised/forPraveer/rsna-pneumonia-detection-challenge/' #change to root dir

sess = tf.InteractiveSession()
input_size = 320
model = get_model(InceptionV3, 0, 7e-5, dropout=None, input_shape=(input_size,input_size,3))

saver = tf.train.Saver()
saver.restore(sess,'../../graphs/' + args.filename) #restore randomized model from filepath

with sess.graph.as_default():
	inp = sess.graph.get_tensor_by_name('input_1:0')
	logits = sess.graph.get_tensor_by_name('dense_1/BiasAdd:0')
	prediction = tf.argmax(logits,1)
	neuron_selector = tf.placeholder(tf.int32)
	y = logits[0][neuron_selector]

train_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/val.csv'
img_list = os.listdir(root_dir+'stage_2_train_images_PNG_resize320_para')
df = pd.read_csv(root_dir + 'stage_2_train_labels.csv')

bb_coords = []
bb_coords2 = []
counts = []
counts2 = []

test_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/test.csv'
#read .csv file indicating
dftest = pd.read_csv(test_labels)
dftest = dftest.drop_duplicates(subset=['name'])

X_test = dftest[['name']]
y_test = dftest[['label']]

X_test.columns = ['name']
y_test.columns = ['label']

test_df = pd.concat([X_test,y_test],axis=1)

X_test = np.asarray([imageio.imread(os.path.join(root_dir + 'stage_2_train_images_PNG_resize320_para/', "{}".format(_))) for _ in test_df.name])
X_test = np.expand_dims(X_test, axis=-1)

X_test = preprocess_input(X_test, 'inception')
X_test2 = np.concatenate((X_test,X_test,X_test),axis=3)

test_ids = np.asarray(list(test_df["name"]))
y_test = np.asarray(list(test_df["label"]))

df = pd.read_csv(root_dir + 'stage_2_train_labels.csv')

scores = defaultdict(list)

mask = np.zeros((len(X_test2))) #select 100 images from the test set for randomization experiments
indices = []
cnt = 0
for i in range(len(X_test2)):
	if y_test[i] == 1:
		indices.append(i)
		cnt += 1

random.shuffle(indices)

for i in range(100):
	mask[indices[i]] = 1


cnt = 0
for j,img in enumerate(X_test2[int(args.start):int(args.end)]): 
	i = j+int(args.start)

	if y_test[i] == 1:
		bb_coords = []
		idx = test_ids[i][:-4]
		df_bb = df[df['patientId']==idx]
		for k in range(0,len(df_bb['patientId'])):
			bb_sub = np.zeros((1,4))
			bb_sub[0,0] = downsamp_coord2(int(df_bb['x'].iloc[k]))
			bb_sub[0,1] = downsamp_coord2(int(df_bb['y'].iloc[k]))
			bb_sub[0,2] = downsamp_coord2(int(df_bb['x'].iloc[k] + df_bb['width'].iloc[k]))
			bb_sub[0,3] = downsamp_coord2(int(df_bb['y'].iloc[k] + df_bb['height'].iloc[k]))
			bb_coords = bb_coords + list(bb_sub)
	
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


if args.filename[-6] == 'n': #output path suffix
	ix = 0 
elif args.filename[-7] == '_':
	ix = int(args.filename[-6])
else:
	ix = 10 + int(args.filename[-6])

output_dir = '../../scores4' #set output dir for saving masks 


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


