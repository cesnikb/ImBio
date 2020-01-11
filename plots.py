import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./metrics/log.csv')
df.plot(x='epoch',figsize=(8,8)).get_figure()
plt.show()

print(df[['Train_auroc','Test_auroc']].max())
