#Exercise 4:
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

california_housing = fetch_california_housing(as_frame=True)

columns_drop = ["Longitude", "Latitude"]
df = california_housing.frame.iloc[::10, :].drop(columns=columns_drop)

sns.pairplot(data = df, hue = 'MedHouseVal')
plt.show()


