import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../plots/auprc_scores_segmentation_withrandom.csv')
column_names = ['patientIdx','Method','trained','AUPRC']
df_long = pd.DataFrame(columns = column_names)
df_np = df.to_numpy()
df_unet = np.load('../plots/temp.npy') # computed UNet scores
mask = np.load('../plots/siim_mask.npy') 
print(len(mask))
df_myscores = pd.read_csv('../plots/auprc_scores_segmentation.csv')
df_myscores = df_myscores.to_numpy()


for i in column_names:
    df_long[i] = np.empty((len(df_myscores)*10 + 800, 0)).tolist()

sal_list = ['GRAD','SG','IG','SIG','GCAM','XRAI','GBP','GGCAM']
cnt,cnt_df = 0,0

unet_avg,avg_avg = 0,0
asums,usums = [],[]

for i in range(8):
    print(np.mean(df_myscores[:,i+1]))
    print(np.std(df_myscores[:,i+1]))
exit()

for i in range(len(df_myscores)):
    for j in range(8):
        df_long['patientIdx'][cnt] = i
        df_long['Method'][cnt] = sal_list[j]
        df_long['trained'][cnt] = 'Trained'
        df_long['AUPRC'][cnt] = df_myscores[i,j+1]
        cnt += 1
    if i < 100:
        print(cnt_df)
        for j in range(8):
            df_long['patientIdx'][cnt] = i
            df_long['Method'][cnt] = sal_list[int(j%8)]
            df_long['trained'][cnt] = 'Untrained'
            df_long['AUPRC'][cnt] = df_np[cnt_df,j + 9]
            cnt += 1
        cnt_df += 1
    df_long['patientIdx'][cnt] = i
    df_long['Method'][cnt] = 'UNET'
    df_long['trained'][cnt] = 'Trained'
    df_long['AUPRC'][cnt] = df_unet[i]
    unet_avg += df_long['AUPRC'][cnt]
    cnt += 1
    df_long['patientIdx'][cnt] = i
    df_long['Method'][cnt] = 'AVG'
    df_long['trained'][cnt] = 'Untrained'
    df_long['AUPRC'][cnt] = df_myscores[i,-1]
    avg_avg += df_long['AUPRC'][cnt]
    asums.append(df_long['AUPRC'][cnt])
    cnt += 1



unet_mean = np.mean(df_unet)
avg_mean = np.mean(df_myscores[:,-1])

df_long['AUPRC'] = df_long['AUPRC'].astype(float)

ax = sns.boxplot(x = 'Method',y = 'AUPRC',data = df_long, hue='trained')
plt.axhline(y=unet_mean, color='r', linestyle='--',label='UNet')
plt.axhline(y=avg_mean, color='r', linestyle=':',label='Average Mask')
plt.ylim((0.,1.))
plt.legend(loc=2,ncol=2,fancybox=True,fontsize='small',framealpha=0.8)
plt.title('Segmentation Utility')
plt.savefig('../plots/naturemi_plots/boxplot_segmentation.png',dpi=300)