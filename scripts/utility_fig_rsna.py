import numpy as np
import os
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

def downsamp_coord2(coord): #downsample coordinates (if required)
  coord_new = coord/320. 
  coord_new = int(320*coord_new)
  return coord_new

def rint(mask):
	mask[mask != 0] = 1
	return mask

def aupr(smap,bb_coords1):
	bb_mask = np.zeros(smap.shape)
	for bbox in bb_coords1:
		x_min = min(int(bbox[0]),int(bbox[2]))
		y_min = min(int(bbox[1]),int(bbox[3]))
		x_max = max(int(bbox[0]),int(bbox[2]))
		y_max = max(int(bbox[1]),int(bbox[3]))
		bb_mask[y_min:(y_max+1),x_min:(x_max+1)] = 1 	
	fpr,tpr,thresholds = precision_recall_curve(rint(bb_mask.flatten()),smap.flatten())
	roc_auc = auc(tpr,fpr)
	return roc_auc

root_dir = '/data/2015P002510/praveer/WeaklySupervised/forPraveer/rsna-pneumonia-detection-challenge/'
mehak_dir = '/data/2015P002510/Mehak/rndetector/Models/Pneumonia/v3/retinanet_4/test_pneumonia/'
test_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/test.csv'
train_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/train.csv'
train_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/train.csv'


dftest = pd.read_csv(test_labels)
dftest = dftest.drop_duplicates(subset=['name'])

X_test = dftest[['name']]
y_test = dftest[['label']]

X_test.columns = ['name']
y_test.columns = ['label']

test_df = pd.concat([X_test,y_test],axis=1)
test_ids = np.asarray(list(test_df["name"]))
y_test = np.asarray(list(test_df["label"]))

dftrain = pd.read_csv(train_labels)
dftrain = dftrain.drop_duplicates(subset=['name'])
X_train = dftest[['name']]
y_train = dftest[['label']]

X_train.columns = ['name']
y_train.columns = ['label']

train_df = pd.concat([X_train,y_train],axis=1)
train_ids = np.asarray(list(test_df["name"]))
y_train = np.asarray(list(train_df["label"]))


dfval = pd.read_csv(val_labels)
dfval = dfval.drop_duplicates(subset=['name'])
X_val = dftest[['name']]
y_val = dftest[['label']]

X_val.columns = ['name']
y_val.columns = ['label']

val_df = pd.concat([X_val,y_val],axis=1)
val_ids = np.asarray(list(val_df["name"]))
y_val = np.asarray(list(val_df["label"]))

retina_masks = []
test_imgs = []

df = pd.read_csv(root_dir + 'stage_2_train_labels.csv')
mask = np.load('../../plots/rsna_mask.npy')

for i in range(200):
	if y_test[i] == 1 and mask[i] == 1:
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
		retina_mask = np.load(mehak_dir+idx+'.npy')
		retina_mask = cv2.resize(retina_mask, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
		# exit()		
		retina_mask = (retina_mask-np.min(retina_mask))/(np.max(retina_mask) - np.min(retina_mask) + 1e-6)
		retina_masks.append(retina_mask)

mask_avg = np.zeros((320,320))

for i in range(X_train): #compute average mask
	if y_train[i] == 1:
		bb_coords = []
		idx = train_ids[i][:-4]
		df_bb = df[df['patientId']==idx]
		for k in range(0,len(df_bb['patientId'])):
			bb_sub = np.zeros((1,4))
			bb_sub[0,0] = downsamp_coord2(int(df_bb['x'].iloc[k]))
			bb_sub[0,1] = downsamp_coord2(int(df_bb['y'].iloc[k]))
			bb_sub[0,2] = downsamp_coord2(int(df_bb['x'].iloc[k] + df_bb['width'].iloc[k]))
			bb_sub[0,3] = downsamp_coord2(int(df_bb['y'].iloc[k] + df_bb['height'].iloc[k]))
			bb_coords = bb_coords + list(bb_sub)

		mask_avg += bb_mask

for i in range(X_val): #compute average mask 
	if y_test[i] == 1: 
		bb_coords = []
		idx = train_ids[i][:-4]
		df_bb = df[df['patientId']==idx]
		for k in range(0,len(df_bb['patientId'])):
			bb_sub = np.zeros((1,4))
			bb_sub[0,0] = downsamp_coord2(int(df_bb['x'].iloc[k]))
			bb_sub[0,1] = downsamp_coord2(int(df_bb['y'].iloc[k]))
			bb_sub[0,2] = downsamp_coord2(int(df_bb['x'].iloc[k] + df_bb['width'].iloc[k]))
			bb_sub[0,3] = downsamp_coord2(int(df_bb['y'].iloc[k] + df_bb['height'].iloc[k]))
			bb_coords = bb_coords + list(bb_sub)

		mask_avg += bb_mask

mask_avg /= len(grad_scores)
		
'''
Load maps
'''

grad_scores = np.load('../../nishanth5/gradient_scores_0.npy')
sg_scores = np.load('../../nishanth5/smoothgradient_scores_0.npy')
ig_scores = np.load('../../nishanth5/ig_scores_0.npy')
sig_scores = np.load('../../nishanth5/smoothig_scores_0.npy')
gradcam_scores = np.load('../../nishanth5/gradcam_scores_0.npy')
xrai_scores = np.load('../../nishanth5/xrai_scores_0.npy')
gbp_scores = np.load('../../nishanth5/gbp_scores_0.npy')
ggcam_scores = np.load('../../nishanth5/ggcam_scores_0.npy')



ratios = []

grad_auc,sg_auc,ig_auc,sig_auc,gradcam_auc,xrai_auc,gbp_auc,ggcam_auc = 0.,0.,0.,0.,0.,0.,0.,0.

grad,sg,ig,sig,gradcam,xrai,gbp,ggcam = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
random_bbox = []

'''
Compute AUPRC
'''

for i in range(len(sg_scores)):
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


r = defaultdict(list)
r['grad'] = grad['AUPRC']
r['sg'] = sg['AUPRC']
r['ig'] = ig['AUPRC']
r['sig'] = sig['AUPRC']
r['gradcam'] = gradcam['AUPRC']
r['xrai'] = xrai['AUPRC']
r['gbp'] = gbp['AUPRC']
r['ggcam'] = ggcam['AUPRC']
r['avg_mask'] = random_bbox

fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(4,2.5))

indices = [11]
idx=indices[0]

i=0

'''
Make utility plot
'''

for idx in indices:
	bbox = xrai_scores[idx,-8]
	# ax[i,0].imshow((np.load('../plots/rsna_npys/{}.npy'.format(idx)) + 1.)/2.)
	ax[0,0].imshow(grad_scores[idx,-7],cmap='plasma')
	ax[0,1].imshow(sg_scores[idx,-7],cmap='plasma')
	ax[0,2].imshow(ig_scores[idx,-7],cmap='plasma')
	ax[0,3].imshow(sig_scores[idx,-7],cmap='plasma')
	ax[1,0].imshow(gradcam_scores[idx,-7],cmap='plasma')
	ax[1,1].imshow(xrai_scores[idx,-7],cmap='plasma')
	ax[1,2].imshow(gbp_scores[idx,-7],cmap='plasma')
	ax[1,3].imshow(ggcam_scores[idx,-7],cmap='plasma')

	ax[0,0].set_title(round(grad['AUPRC'][idx],2),pad=3,size=7)
	ax[0,1].set_title(round(sg['AUPRC'][idx],2),pad=3,size=7)
	ax[0,2].set_title(round(ig['AUPRC'][idx],2),pad=3,size=7)
	ax[0,3].set_title(round(sig['AUPRC'][idx],2),pad=3,size=7)
	ax[1,0].set_title(round(gradcam['AUPRC'][idx],2),pad=3,size=7)
	ax[1,1].set_title(round(xrai['AUPRC'][idx],2),pad=3,size=7)
	ax[1,2].set_title(round(gbp['AUPRC'][idx],2),pad=3,size=7)
	ax[1,3].set_title(round(ggcam['AUPRC'][idx],2),pad=3,size=7)

	for i in range(2):
		for j in range(4):
			ax[i,j].axis('off')
			for k in range(len(bbox)):
				ax[i,j].add_patch(Rectangle((downsamp_coord2(bbox[k][0]),downsamp_coord2(bbox[k][1])),downsamp_coord2(bbox[k][2]) - downsamp_coord2(bbox[k][0]),downsamp_coord2(bbox[k][3]) - downsamp_coord2(bbox[k][1]),fill=False,edgecolor='black'))

plt.savefig('../../plots/naturemi_plots/utility_rsna_1.png',dpi=300)

plt.close('all')

bbox = xrai_scores[idx,-8]

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(3,2))
ax[0].imshow((np.load('../../plots/rsna_npys/{}.npy'.format(idx)) + 1.)/2.)
ax[1].imshow(retina_masks[idx],cmap = 'plasma')
ax[1].set_title(round(aupr(retina_masks[idx],xrai_scores[idx,-8]),3),pad=3,size=7)
ax[2].imshow(mask_avg,cmap = 'plasma')
ax[2].set_title(round(random_bbox[idx],3),pad=3,size=7)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
for i in range(3):
	for k in range(len(bbox)):
		ax[i].add_patch(Rectangle((downsamp_coord2(bbox[k][0]),downsamp_coord2(bbox[k][1])),downsamp_coord2(bbox[k][2]) - downsamp_coord2(bbox[k][0]),downsamp_coord2(bbox[k][3]) - downsamp_coord2(bbox[k][1]),fill=False,edgecolor='black'))
plt.savefig('../../plots/naturemi_plots/utility_rsna_2.png',dpi=300)
# ax[1].imshow(retina_masks[5],cmap = 'plasma')






