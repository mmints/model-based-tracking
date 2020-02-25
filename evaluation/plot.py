import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

delta = pd.read_csv('normal.csv')

d_x = delta[['Frame ID', 'X']]
d_y = delta[['Frame ID', 'Y']]
d_z = delta[['Frame ID', 'Z']]

chart_x = sns.scatterplot(x="Frame ID", y="X", data=d_x) # blue
chart_y = sns.scatterplot(x="Frame ID", y="Y", data=d_y) # orange
chart_z = sns.scatterplot(x="Frame ID", y="Z", data=d_z) # green

plt.show()