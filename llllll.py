import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
df = pd.read_csv('taxi.csv')
df.info()
df.describe()

%matplotlib inline
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.scatter(df.distance_traveled,df.fare,color='red',marker='+')
     

