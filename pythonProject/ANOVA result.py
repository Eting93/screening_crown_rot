import pandas as pd
import scipy.stats as stats
import researchpy as rp

df = pd.read_excel("D:\crown_rot_image\colour\manual_cut_28_07_result/test.xlsx",sheet_name='Sheet2')
df.drop('Plant', axis= 1, inplace= True)

# Recoding value from numeric to string
df['Disease'].replace({1: 'Infected', 2: 'Control'}, inplace= True)

df.info()
result = rp.summary_cont(df['hist_h3'])
print(result)




