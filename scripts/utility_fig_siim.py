import numpy as np
import os, random
random.seed(1)
import matplotlib as mpl
import scipy.stats
from matplotlib.patches import Rectangle
from collections import defaultdict
from sklearn.model_selection import train_test_split
mpl.rc('figure', max_open_warning = 0)
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from scipy.stats import wasserstein_distance
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,auc,f1_score,average_precision_score

def rint(mask): #binarize mask based on nonzero pixels
	mask[mask != 0] = 1
	return mask

def aupr(smap,bb_mask):
	fpr,tpr,thresholds = precision_recall_curve(rint(bb_mask.flatten()),smap.flatten())
	roc_auc = auc(tpr,fpr)
	return roc_auc

train_labels = '/data/2015P002510/Bryan/Pneumothorax_Data/csvs/train-rle.csv' #train file for the SIIM ACR pneumothorax dataset 
unet_masks_dir = '/data/2015P002510/Bryan/Pytorch/Segmentation/im_results_nish/' #test set masks produced by the UNet
root_dir = '/data/2015P002510/Bryan/Pneumothorax_Data/PNGs/' 
mask_dir = root_dir + 'mask/'

df = pd.read_csv(train_labels)
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
x_train = np.squeeze(np.array(x_train))
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.1,random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=1)

unet_masks = [] #Store baseline UNET masks
test_imgs = [] #Store reference images (100) for ease of plotting
test_masks = []
unet_auprs = []

for i,img in enumerate(x_test):
	if y_test[i] == 1 and i < 200:
		unet_masks.append(cv2.imread(unet_masks_dir+img+'.png'))
		test_imgs.append(cv2.imread(root_dir+'train/'+img+'.png'))
		test_masks.append(cv2.imread(root_dir+'mask/'+img+'.png'))
		unet_auprs.append(aupr(unet_masks[-1],test_masks[-1]))


grad_scores = np.load('../../scores1/gradient_scores_0.npy')
sg_scores = np.load('../../scores1/smoothgradient_scores_0.npy')
ig_scores = np.load('../../scores1/ig_scores_0.npy')
sig_scores = np.load('../../scores1/smoothig_scores_0.npy')
gradcam_scores = np.load('../../scores1/gradcam_scores_0.npy')
xrai_scores = np.load('../../scores1/xrai_scores_0.npy')
gbp_scores = np.load('../../scores1/gbp_scores_0.npy')
ggcam_scores = np.load('../../scores1/ggcam_scores_0.npy')

print('Loaded saliency maps')

mask_avg = np.zeros((320,320))

for i,img in enumerate(x_train): #compute average mask based on training and val sets
	mask = cv2.imread(os.path.join(mask_dir,img,'.png'))
	mask_avg += mask

for i,img in enumerate(x_val):
	mask = cv2.imread(os.path.join(mask_dir,img,'.png'))
	mask_avg += mask

mask_avg /= (len(x_train) + len(x_val))

grad,sg,ig,sig,gradcam,xrai,gbp,ggcam = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
random_bbox = [] #store average mask scores

for i in range(100):
	if i%10 == 0:
		print(i)
	bbox = grad_scores[i,-8]
	grad['AUPRC'].append(aupr(grad_scores[i,-7],bbox))
	sg['AUPRC'].append(aupr(sg_scores[i,-7],bbox))
	ig['AUPRC'].append(aupr(ig_scores[i,-7],bbox))
	sig['AUPRC'].append(aupr(sig_scores[i,-7],bbox))
	gradcam['AUPRC'].append(aupr(gradcam_scores[i,-7],bbox))
	xrai['AUPRC'].append(aupr(xrai_scores[i,-7],bbox))
	gbp['AUPRC'].append(aupr(gbp_scores[i,-7],bbox))
	ggcam['AUPRC'].append(aupr(ggcam_scores[i,-7],bbox))
	random_bbox.append(aupr(mask_avg,bbox))

fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(4,2.5))
indices = [10,21] #sample from the first 100

# 10, 15

unet_masks = np.array(unet_masks).astype(float)

i = 0
idx = 17

ax[0,0].imshow(grad_scores[idx,-7],cmap='plasma',alpha=1.)
ax[0,1].imshow(sg_scores[idx,-7],cmap='plasma',alpha=1.)
ax[0,2].imshow(ig_scores[idx,-7],cmap='plasma',alpha=1.)
ax[0,3].imshow(sig_scores[idx,-7],cmap='plasma',alpha=1.)
ax[1,0].imshow(gradcam_scores[idx,-7],cmap='plasma',alpha=1.)
ax[1,1].imshow(xrai_scores[idx,-7],cmap='plasma',alpha=1.)
ax[1,2].imshow(gbp_scores[idx,-7],cmap='plasma',alpha=1.)
ax[1,3].imshow(ggcam_scores[idx,-7],cmap='plasma',alpha=1.)

ax[0,0].set_title(round(grad['AUPRC'][idx],3),pad=1,fontsize='small')
ax[0,1].set_title(round(sg['AUPRC'][idx],3),pad=1,fontsize='small')
ax[0,2].set_title(round(ig['AUPRC'][idx],3),pad=1,fontsize='small')
ax[0,3].set_title(round(sig['AUPRC'][idx],3),pad=1,fontsize='small')
ax[1,0].set_title(round(gradcam['AUPRC'][idx],3),pad=1,fontsize='small')
ax[1,1].set_title(round(xrai['AUPRC'][idx],3),pad=1,fontsize='small')
ax[1,2].set_title(round(gbp['AUPRC'][idx],3),pad=1,fontsize='small')
ax[1,3].set_title(round(ggcam['AUPRC'][idx],3),pad=1,fontsize='small')

for i in range(2):
	for j in range(4):
		ax[i,j].axis('off')

plt.savefig('../../plots/naturemi_plots/utility_siim1.png',dpi=300)

plt.clf()

fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(4,1.5))

ax[0].imshow(test_imgs[idx])
ax[1].imshow(test_masks[idx])

ax[0].axis('off')
ax[1].axis('off')
ax[2].imshow(mask_avg,cmap='plasma')
ax[3].imshow(unet_masks[idx,:,:,0],cmap='plasma')

ax[2].axis('off')
ax[3].axis('off')

ax[2].set_title(round(random_bbox[idx],3),pad=1,fontsize='small')
ax[3].set_title(round(aupr(unet_masks[idx,:,:,0],grad_scores[idx,-8]),3),pad=1,fontsize='small')

plt.savefig('../../plots/naturemi_plots/utility_siim2.png',dpi=300)



