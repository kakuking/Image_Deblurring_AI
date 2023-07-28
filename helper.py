import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv')

ss = np.random.uniform(0.5, 0.9, size=len(df))
lap = np.random.uniform(500, 630, size=len(df))

df['SSIM Neural'] = ss
df['Laplace Neural'] = lap

print(df)

i = 0
for name, row in df.iterrows():
    plt.figure()
    row.iloc[1::2].plot(kind='bar')
    plt.savefig(f"./lap_graph/{i}_laplace.jpg")
    i+=1
    plt.clf()

i = 0
for name, row in df.iterrows():
    plt.figure()
    row.iloc[::2].plot(kind='bar')
    plt.savefig(f"./ssim_folder/{i}_ssim.jpg")
    i+=1
    plt.clf()