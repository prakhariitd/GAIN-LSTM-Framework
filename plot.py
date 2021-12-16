import pandas as pd
import matplotlib.pyplot as plt

filepath = 'data/pmu.csv'
series = pd.read_csv(filepath, parse_dates=False, index_col=None, usecols = [20])
series.plot()
plt.show()