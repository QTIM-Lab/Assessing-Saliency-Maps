import numpy as np
from collections import defaultdict
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,cv2

def corr(maps1,maps2): #Spearman rank correlation
	scores = []
	assert len(maps1) == len(maps2)
	for m1,m2 in zip(maps1,maps2):
		scores.append(spearmanr(m1.flatten(),m2.flatten())[0])
	return scores

def ssim1(maps1,maps2): #SSIM
	scores = []
	assert len(maps1) == len(maps2)
	for m1,m2 in zip(maps1,maps2): 	
		scores.append(ssim(m1,m2))
	return scores


'''Load RetinaNet masks for baseline computation'''

root_dir = '/data/2015P002510/praveer/WeaklySupervised/forPraveer/rsna-pneumonia-detection-challenge/'
mehak_dir = '/data/2015P002510/Mehak/rndetector/Models/Pneumonia/v3/retinanet_4/test_pneumonia/'
mehak_dir2 = '/data/2015P002510/Mehak/rndetector/Models/Pneumonia/v4/retinanet_2/test_pneumonia/'
test_labels = root_dir + 'CSV_files_stage_2_train_labels_with_2_classes/test.csv'
dftest = pd.read_csv(test_labels)
dftest = dftest.drop_duplicates(subset=['name'])

X_test = dftest[['name']]
y_test = dftest[['label']]

X_test.columns = ['name']
y_test.columns = ['label']

test_df = pd.concat([X_test,y_test],axis=1)
test_ids = np.asarray(list(test_df["name"]))
y_test = np.asarray(list(test_df["label"]))

retina_masks,retina_masks2 = [],[]
test_imgs = []

for i in range(len(os.listdir(mehak_dir))):
	if y_test[i] == 1:
		idx = test_ids[i][:-4]
		retina_mask = np.load(mehak_dir+idx+'.npy')
		retina_mask = cv2.resize(retina_mask, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
		retina_mask2 = np.load(mehak_dir2+idx+'.npy')
		retina_mask2 = cv2.resize(retina_mask2, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
	
		retina_masks.append(retina_mask)
		retina_masks2.append(retina_mask2)


repeatability = defaultdict(list)
maps = defaultdict(list)
reproducibility = defaultdict(list)

for i in tqdm(range(3)):
	maps['grad'].append(np.load('../../nishanth{}/gradient_scores_0.npy'.format(i+6)))
	maps['sg'].append(np.load('../../nishanth{}/smoothgradient_scores_0.npy'.format(i+6)))
	maps['ig'].append(np.load('../../nishanth{}/ig_scores_0.npy'.format(i+6)))
	maps['sig'].append(np.load('../../nishanth{}/smoothig_scores_0.npy'.format(i+6)))
	maps['gcam'].append(np.load('../../nishanth{}/gradcam_scores_0.npy'.format(i+6)))
	maps['xrai'].append(np.load('../../nishanth{}/xrai_scores_0.npy'.format(i+6)))
	maps['gbp'].append(np.load('../../nishanth{}/gbp_scores_0.npy'.format(i+6)))
	maps['ggcam'].append(np.load('../../nishanth{}/ggcam_scores_0.npy'.format(i+6)))


for key in maps.keys():
	repeatability[key] = ssim1(maps[key][1][:,-7],maps[key][0][:,-7])
	reproducibility[key] = ssim1(maps[key][2][:,-7],maps[key][1][:,-7])
	print(key, ' ', np.mean(repeatability[key]), ' ', np.mean(reproducibility[key]))
	print(key, ' ', np.std(repeatability[key]), ' ', np.std(reproducibility[key]))

rnet_baseline = np.mean(ssim1(retina_masks, retina_masks2))
print(rnet_baseline)

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

plt.axhline(rnet_baseline,linestyle='--')
plt.savefig('../../plots/naturemi_plots/repeatability_rsna.png',dpi=300)







