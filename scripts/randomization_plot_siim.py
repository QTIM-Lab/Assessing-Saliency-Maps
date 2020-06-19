import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def mean_without_nan(l): #Ignore NaNs while computing mean/std
	if np.isnan(np.mean(l)) == False:
		return np.mean(l)
	elif np.isnan(np.mean([l[0],l[1]])) == False:
		return np.mean([l[0],l[1]])
	elif np.isnan(np.mean([l[0],l[2]])) == False:
		return np.mean([l[0],l[2]])
	elif np.isnan(np.mean([l[1],l[2]])) == False:
		return np.mean([l[1],l[2]])
	elif np.isnan(l[0]) == False:
		return l[0]
	elif np.isnan(l[1]) == False:
		return l[1]
	elif np.isnan(l[2]) == False:
		return l[2]
	else:
		print('NaNs detected, returning 0')
		return 0

def std_without_nan(l):
	if np.isnan(np.std(l)) == False:
		return np.std(l)
	elif np.isnan(np.std([l[0],l[1]])) == False:
		return np.std([l[0],l[1]])
	elif np.isnan(np.std([l[0],l[2]])) == False:
		return np.std([l[0],l[2]])
	elif np.isnan(np.std([l[1],l[2]])) == False:
		return np.std([l[1],l[2]])
	elif np.isnan(l[0]) == False:
		return 0
	elif np.isnan(l[1]) == False:
		return 0
	elif np.isnan(l[2]) == False:
		return 0
	else:
		print('NaNs detected, returning 0')
		return 0



def ssim1(maps1,maps2): #SSIM
	scores = []
	if np.shape(maps1) == (320,320):
		return ssim(maps1,maps2)
	if len(maps1) != len(maps2):
		maps1 = maps1[:97]
	for m1,m2 in zip(maps1,maps2): 	
		scores.append(ssim(m1,m2))
	return [x for x in scores if np.isnan(x) == False]


scores1 = defaultdict(list)
scores2 = defaultdict(list)
scores3 = defaultdict(list)

dir1 = '../../nishanth4/' #change to directories with computed maps
dir2 = '../../scores_ptx_casc/'
dir3 = '../../scores_ptx_casc_repl/'



for i in tqdm(range(12)):
	scores1['grad'].append(np.load(os.path.join(dir1,'gradient_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['grad'].append(np.load(os.path.join(dir2,'gradient_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['grad'].append(np.load(os.path.join(dir3,'gradient_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['sg'].append(np.load(os.path.join(dir1,'smoothgradient_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['sg'].append(np.load(os.path.join(dir2,'smoothgradient_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['sg'].append(np.load(os.path.join(dir3,'smoothgradient_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['ig'].append(np.load(os.path.join(dir1,'ig_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['ig'].append(np.load(os.path.join(dir2,'ig_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['ig'].append(np.load(os.path.join(dir3,'ig_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['sig'].append(np.load(os.path.join(dir1,'smoothig_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['sig'].append(np.load(os.path.join(dir2,'smoothig_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['sig'].append(np.load(os.path.join(dir3,'smoothig_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['gcam'].append(np.load(os.path.join(dir1,'gradcam_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['gcam'].append(np.load(os.path.join(dir2,'gradcam_scores_{}.npy'.format(i)),allow_pickle=True))
	try:
		scores3['gcam'].append(np.load(os.path.join(dir3,'gradcam_scores_{}.npy'.format(i)),allow_pickle=True))
	except:
		scores3['gcam'].append(np.array([np.nan]*100))

	scores1['xrai'].append(np.load(os.path.join(dir1,'xrai_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['xrai'].append(np.load(os.path.join(dir2,'xrai_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['xrai'].append(np.load(os.path.join(dir3,'xrai_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['gbp'].append(np.load(os.path.join(dir1,'gbp_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['gbp'].append(np.load(os.path.join(dir2,'gbp_scores_{}.npy'.format(i)),allow_pickle=True))
	scores3['gbp'].append(np.load(os.path.join(dir3,'gbp_scores_{}.npy'.format(i)),allow_pickle=True))

	scores1['ggcam'].append(np.load(os.path.join(dir1,'ggcam_scores_{}.npy'.format(i)),allow_pickle=True))
	scores2['ggcam'].append(np.load(os.path.join(dir2,'ggcam_scores_{}.npy'.format(i)),allow_pickle=True))
	try:
		scores3['ggcam'].append(np.load(os.path.join(dir3,'ggcam_scores_{}.npy'.format(i)),allow_pickle=True))
	except:
		scores3['ggcam'].append(np.array([np.nan]*100))


agg_scores1 = defaultdict(list)
agg_scores2 = defaultdict(list)
agg_scores3 = defaultdict(list)

mean_scores = defaultdict(list)
std_scores = defaultdict(list)


for i in tqdm(range(12)):
	agg_scores1['grad'].append(np.mean(ssim1(scores1['grad'][i][:,-7],scores1['grad'][0][:,-7])))
	agg_scores2['grad'].append(np.mean(ssim1(scores2['grad'][i][:,-7],scores2['grad'][0][:,-7])))
	agg_scores3['grad'].append(np.mean(ssim1(scores3['grad'][i][:,-7],scores3['grad'][0][:,-7])))
	mean_scores['grad'].append(mean_without_nan([agg_scores1['grad'][i],agg_scores2['grad'][i],agg_scores3['grad'][i]]))
	std_scores['grad'].append(std_without_nan([agg_scores1['grad'][i],agg_scores2['grad'][i],agg_scores3['grad'][i]]))


	agg_scores1['sg'].append(np.mean(ssim1(scores1['sg'][i][:,-7],scores1['sg'][0][:,-7])))
	agg_scores2['sg'].append(np.mean(ssim1(scores2['sg'][i][:,-7],scores2['sg'][0][:,-7])))
	agg_scores3['sg'].append(np.mean(ssim1(scores3['sg'][i][:,-7],scores3['sg'][0][:,-7])))
	mean_scores['sg'].append(mean_without_nan([agg_scores1['sg'][i],agg_scores2['sg'][i],agg_scores3['sg'][i]]))
	std_scores['sg'].append(std_without_nan([agg_scores1['sg'][i],agg_scores2['sg'][i],agg_scores3['sg'][i]]))

	agg_scores1['ig'].append(np.mean(ssim1(scores1['ig'][i][:,-7],scores1['ig'][0][:,-7])))
	agg_scores2['ig'].append(np.mean(ssim1(scores2['ig'][i][:,-7],scores2['ig'][0][:,-7])))
	agg_scores3['ig'].append(np.mean(ssim1(scores3['ig'][i][:,-7],scores3['ig'][0][:,-7])))
	mean_scores['ig'].append(mean_without_nan([agg_scores1['ig'][i],agg_scores2['ig'][i],agg_scores3['ig'][i]]))
	std_scores['ig'].append(std_without_nan([agg_scores1['ig'][i],agg_scores2['ig'][i],agg_scores3['ig'][i]]))

	agg_scores1['sig'].append(np.mean(ssim1(scores1['sig'][i][:,-7],scores1['sig'][0][:,-7])))
	agg_scores2['sig'].append(np.mean(ssim1(scores2['sig'][i][:,-7],scores2['sig'][0][:,-7])))
	agg_scores3['sig'].append(np.mean(ssim1(scores3['sig'][i][:,-7],scores3['sig'][0][:,-7])))
	mean_scores['sig'].append(mean_without_nan([agg_scores1['sig'][i],agg_scores2['sig'][i],agg_scores3['sig'][i]]))
	std_scores['sig'].append(std_without_nan([agg_scores1['sig'][i],agg_scores2['sig'][i],agg_scores3['sig'][i]]))

	agg_scores1['gcam'].append(np.mean(ssim1(scores1['gcam'][i][:,-7],scores1['gcam'][0][:,-7])))
	agg_scores2['gcam'].append(np.mean(ssim1(scores2['gcam'][i][:,-7],scores2['gcam'][0][:,-7])))
	try:
		agg_scores3['gcam'].append(np.mean(ssim1(scores3['gcam'][i][:,-7],scores3['gcam'][0][:,-7])))
	except:
		agg_scores3['gcam'].append(np.nan)
	mean_scores['gcam'].append(mean_without_nan([agg_scores1['gcam'][i],agg_scores2['gcam'][i],agg_scores3['gcam'][i]]))
	std_scores['gcam'].append(std_without_nan([agg_scores1['gcam'][i],agg_scores2['gcam'][i],agg_scores3['gcam'][i]]))

	agg_scores1['xrai'].append(np.mean(ssim1(scores1['xrai'][i][:,-7],scores1['xrai'][0][:,-7])))
	agg_scores2['xrai'].append(np.mean(ssim1(scores2['xrai'][i][:,-7],scores2['xrai'][0][:,-7])))
	agg_scores3['xrai'].append(np.mean(ssim1(scores3['xrai'][i][:,-7],scores3['xrai'][0][:,-7])))
	mean_scores['xrai'].append(mean_without_nan([agg_scores1['xrai'][i],agg_scores2['xrai'][i],agg_scores3['xrai'][i]]))
	std_scores['xrai'].append(std_without_nan([agg_scores1['xrai'][i],agg_scores2['xrai'][i],agg_scores3['xrai'][i]]))

	agg_scores1['gbp'].append(np.mean(ssim1(scores1['gbp'][i][:,-7],scores1['gbp'][0][:,-7])))
	agg_scores2['gbp'].append(np.mean(ssim1(scores2['gbp'][i][:,-7],scores2['gbp'][0][:,-7])))
	agg_scores3['gbp'].append(np.mean(ssim1(scores3['gbp'][i][:,-7],scores3['gbp'][0][:,-7])))
	mean_scores['gbp'].append(mean_without_nan([agg_scores1['gbp'][i],agg_scores2['gbp'][i],agg_scores3['gbp'][i]]))
	std_scores['gbp'].append(std_without_nan([agg_scores1['gbp'][i],agg_scores2['gbp'][i],agg_scores3['gbp'][i]]))

	agg_scores1['ggcam'].append(np.mean(ssim1(scores1['ggcam'][i][:,-7],scores1['ggcam'][0][:,-7])))
	agg_scores2['ggcam'].append(np.mean(ssim1(scores2['ggcam'][i][:,-7],scores2['ggcam'][0][:,-7])))
	try:
		agg_scores3['ggcam'].append(np.mean(ssim1(scores3['ggcam'][i][:,-7],scores3['ggcam'][0][:,-7])))
	except:
		agg_scores3['ggcam'].append(np.nan)
	mean_scores['ggcam'].append(mean_without_nan([agg_scores1['ggcam'][i],agg_scores2['ggcam'][i],agg_scores3['ggcam'][i]]))
	std_scores['ggcam'].append(std_without_nan([agg_scores1['ggcam'][i],agg_scores2['ggcam'][i],agg_scores3['ggcam'][i]]))


x = ['Original','Logits','10','9','8','7','6','5','4','3','2','1']

'''
Compute baseline for XRAI
'''

baseline_scores = {}

for key in scores1.keys():
	ind_scores = []
	for i in tqdm(range(len(scores1[key][0]))):
		for j in range(50):
			idx1 = np.random.randint(low=0,high=len(scores1[key][0])-1)
			idx2 = np.random.randint(low=0,high=len(scores1[key][0])-1)

			while(idx1 == idx2): #ensure different indices
				idx2 = np.random.randint(low=0,high=len(scores1[key][0])-1)

			ind_scores.append(ssim1(scores1[key][0][idx1,-7],scores1[key][0][idx2,-7]))
	baseline_scores[key] = np.mean(ind_scores)

print(baseline_scores)



plt.plot(x,mean_scores['grad'],'r',label = 'GRAD')
plt.plot(x,mean_scores['sg'],'b',label = 'SG')
plt.plot(x,mean_scores['ig'],'y',label = 'IG')
plt.plot(x,mean_scores['sig'],'g',label = 'SIG')
plt.plot(x,mean_scores['gcam'],'c',label = 'GCAM')
plt.plot(x,mean_scores['xrai'],'m',label = 'XRAI')
plt.plot(x,mean_scores['gbp'],'k',label = 'GBP')
plt.plot(x,mean_scores['ggcam'],'0.5',label = 'GGCAM')

for key in mean_scores.keys():
	print(key, ' ', std_scores[key][-1])

plt.axhline(baseline_scores['grad'],color = 'r', linestyle='--')
plt.axhline(baseline_scores['sg'],color = 'b', linestyle='--')
plt.axhline(baseline_scores['ig'],color = 'y', linestyle='--')
plt.axhline(baseline_scores['sig'],color = 'g', linestyle='--')
plt.axhline(baseline_scores['gcam'],color = 'c', linestyle='--')
plt.axhline(baseline_scores['xrai'],color = 'm', linestyle='--')
plt.axhline(baseline_scores['gbp'],color = 'k', linestyle='--')
plt.axhline(baseline_scores['ggcam'],color = '0.5', linestyle='--')

for key in mean_scores.keys():
	mean_scores[key] = np.array(mean_scores[key])
	std_scores[key] = np.array(std_scores[key])

plt.fill_between(x,mean_scores['grad'] - std_scores['grad'],mean_scores['grad'] + std_scores['grad'],color='r',alpha=0.1)
plt.fill_between(x,mean_scores['sg'] - std_scores['sg'],mean_scores['sg'] + std_scores['sg'],color='b',alpha=0.1)
plt.fill_between(x,mean_scores['ig'] - std_scores['ig'],mean_scores['ig'] + std_scores['ig'],color='y',alpha=0.1)
plt.fill_between(x,mean_scores['sig'] - std_scores['sig'],mean_scores['sig'] + std_scores['sig'],color='g',alpha=0.1)
plt.fill_between(x,mean_scores['gcam'] - std_scores['gcam'],mean_scores['gcam'] + std_scores['gcam'],color='c',alpha=0.1)
plt.fill_between(x,mean_scores['xrai'] - std_scores['xrai'],mean_scores['xrai'] + std_scores['xrai'],color='m',alpha=0.1)
plt.fill_between(x,mean_scores['gbp'] - std_scores['gbp'],mean_scores['gbp'] + std_scores['gbp'],color='k',alpha=0.1)
plt.fill_between(x,mean_scores['ggcam'] - std_scores['ggcam'],mean_scores['ggcam'] + std_scores['ggcam'],color='0.5',alpha=0.1)

plt.xticks(rotation=25)
plt.xlabel('Layer')
plt.ylabel('SSIM')

plt.legend(fontsize = 5,loc='best')

plt.savefig('../../plots/naturemi_plots/randomization_ssim.png',dpi=300)



