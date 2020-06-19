import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../plots/auprc_scores_detection_withrandom.csv')
column_names = ['patientIdx','Method','trained','AUPRC']
df_long = pd.DataFrame(columns = column_names) #long form of the dataframe

df_np = df.to_numpy()

df_rnet = np.load('../plots/temp2.npy') #Computed retinanet AUPRCs
mask = np.load('../plots/rsna_mask.npy')

df_myscores = pd.read_csv('../plots/auprc_scores_detection.csv')

df_myscores = df_myscores.to_numpy()

df_rnet = df_rnet[:-6]

for i in column_names: 
    df_long[i] = np.empty((len(df_myscores)*10, 0)).tolist()

sal_list = ['GRAD','SG','IG','SIG','GCAM','XRAI','GBP','GGCAM']
cnt,cnt_df,cnt_avg = 0,0,0
rsum,asum = 0,0
asums,rsums = [],[]

avg_scores = df_myscores[:,-1]

for i in range(len(df_myscores)):
    for j in range(8):
        df_long['patientIdx'][cnt] = i
        df_long['Method'][cnt] = sal_list[j]
        df_long['trained'][cnt] = 'Trained'
        df_long['AUPRC'][cnt] = df_myscores[i,j+1]
        cnt += 1
    if mask[i]: #Add the random model's score
        print(cnt_df)

        for j in range(8):
            df_long['patientIdx'][cnt] = i
            df_long['Method'][cnt] = sal_list[int(j%8)]
            df_long['trained'][cnt] = 'Untrained'
            df_long['AUPRC'][cnt] = df_np[cnt_df,j + 9]
            cnt += 1
        cnt_df += 1

    df_long['patientIdx'][cnt] = i
    df_long['Method'][cnt] = 'AVG'
    df_long['trained'][cnt] = 'Untrained'
    df_long['AUPRC'][cnt] = avg_scores[i]
    asum += avg_scores[i]
    asums.append(avg_scores[i])
    cnt += 1
    df_long['patientIdx'][cnt] = i
    df_long['Method'][cnt] = 'RNET'
    # print(i)
    df_long['trained'][cnt] = 'Trained'
    df_long['AUPRC'][cnt] = df_rnet[i]
    rsum += df_rnet[i]
    rsums.append(df_rnet[i])
    cnt += 1

rnet_mean = np.mean(df_rnet)
avg_mean = np.mean(asums)

df_long['AUPRC'] = df_long['AUPRC'].astype(float)

ax = sns.boxplot(x = 'Method',y = 'AUPRC',data = df_long, hue='trained',dodge=True)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.axhline(y=rnet_mean, color='r', linestyle='--',label='RetinaNet')
plt.axhline(y=avg_mean, color='r', linestyle=':',label='Average Box')
plt.ylim((0.,1.))
plt.axvline(x=7.5189566,color='k')
plt.legend(loc=2,ncol=2,fancybox=True,fontsize='small',framealpha=0.8)
plt.title('Detection Utility')
plt.savefig('../plots/naturemi_plots/boxplot_detection3.png',dpi=300)


