import numpy as np
from collections import defaultdict
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

def corr(maps1,maps2): #spearman rank correlation
	scores = []
	assert len(maps1) == len(maps2)
	for m1,m2 in zip(maps1,maps2):
		scores.append(spearmanr(m1.flatten(),m2.flatten())[0])
	return scores

def ssim1(maps1,maps2): #SSIM
	scores = []
	if len(maps1) < len(maps2):
		maps2 = maps2[:len(maps1)]
	elif len(maps1) > len(maps2):
		maps1 = maps1[:len(maps2)]
	assert len(maps1) == len(maps2)
	for m1,m2 in zip(maps1,maps2): 	
		scores.append(ssim(m1,m2))
	return scores

repeatability = defaultdict(list)
maps = defaultdict(list)
reproducibility = defaultdict(list)

train_labels = '/data/2015P002510/Bryan/Pneumothorax_Data/csvs/train-rle.csv'
unet_dir1 = '/data/2015P002510/Bryan/Pytorch/Segmentation/im_results_nishv1/'
unet_dir2 = '/data/2015P002510/Bryan/Pytorch/Segmentation/im_results_nishv2/'
root_dir = '/data/2015P002510/Bryan/Pneumothorax_Data/PNGs/'
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

unet_masks, unet_masks2 = [], [] #Store baseline UNET masks
test_imgs = [] #Store reference images (100) for ease of plotting
test_masks = []
unet_auprs = []

for i,img in enumerate(x_test):
	if y_test[i] == 1:
		unet_masks.append(cv2.imread(unet_dir1+img+'.png')[:,:,0])
		unet_masks2.append(cv2.imread(unet_dir2+img+'.png')[:,:,0])
		print(i)

unet_baseline = np.mean(ssim1(unet_masks, unet_masks2))

for i in tqdm(range(2)):
	maps['grad'].append(np.load('../../nishanth{}/gradient_scores_0.npy'.format(i+2)))
	maps['sg'].append(np.load('../../nishanth{}/smoothgradient_scores_0.npy'.format(i+2)))
	maps['ig'].append(np.load('../../nishanth{}/ig_scores_0.npy'.format(i+2)))
	maps['sig'].append(np.load('../../nishanth{}/smoothig_scores_0.npy'.format(i+2)))
	maps['gcam'].append(np.load('../../nishanth{}/gradcam_scores_0.npy'.format(i+2)))
	maps['xrai'].append(np.load('../../nishanth{}/xrai_scores_0.npy'.format(i+2)))
	maps['gbp'].append(np.load('../../nishanth{}/gbp_scores_0.npy'.format(i+2)))
	maps['ggcam'].append(np.load('../../nishanth{}/ggcam_scores_0.npy'.format(i+2)))

maps['grad'].append(np.load('../../scores_ptx_casc/gradient_scores_0.npy'))
maps['sg'].append(np.load('../../scores_ptx_casc/smoothgradient_scores_0.npy'))
maps['ig'].append(np.load('../../scores_ptx_casc/ig_scores_0.npy'))
maps['sig'].append(np.load('../../scores_ptx_casc/smoothig_scores_0.npy'))
maps['gcam'].append(np.load('../../scores_ptx_casc/gradcam_scores_0.npy'))
maps['xrai'].append(np.load('../../scores_ptx_casc/xrai_scores_0.npy'))
maps['gbp'].append(np.load('../../scores_ptx_casc/gbp_scores_0.npy'))
maps['ggcam'].append(np.load('../../scores_ptx_casc/ggcam_scores_0.npy'))

for key in maps.keys():
	repeatability[key] = ssim1(maps[key][1][:,-7],maps[key][0][:,-7])
	reproducibility[key] = ssim1(maps[key][2][:,-7],maps[key][1][:,-7])
	print(key, ' ', np.mean(repeatability[key]), ' ', np.mean(reproducibility[key]))
	print(key, ' ', np.std(repeatability[key]), ' ', np.std(reproducibility[key]))

df_vals, df_keys1, df_keys2 = [],[],[]

for key in maps.keys():
	for i in range(len(repeatability[key])):
		df_vals.append(repeatability[key][i])
		df_keys1.append(key.upper())
		df_keys2.append('Repeatability')

	for i in range(len(reproducibility[key])):
		df_vals.append(reproducibility[key][i])
		df_keys1.append(key.upper())
		df_keys2.append('Reproducibility')

df_new = list(zip(df_vals,df_keys1,df_keys2))
df_new_pd = pd.DataFrame(data=df_new,index=None,columns=['SSIM','Saliency Map','Type'])

ax = sns.boxplot(x="Saliency Map", y = "SSIM",data = df_new_pd, palette=sns.color_palette("RdBu", 2),hue = 'Type')

plt.ylim((0,1.))
plt.axhline(unet_baseline,linestyle='--')

plt.savefig('../../plots/naturemi_plots/repeatability_siim.png',dpi=300)







