#Exercise 3:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly
import plotly.graph_objs as go

df = pd.read_csv('vehicles.csv').drop('make', axis = 1)
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
reg = LinearRegression().fit(x_scaled, y)
print(f'The weighted coefficients are: \n{reg.coef_}')
coefficients = np.abs(reg.coef_)

weight_coefficients = []
for i in range(5):
    most_weight = np.argmax(coefficients)
    weight_coefficients.append(df.columns[most_weight + 1])
    coefficients[most_weight] = -np.inf

print(f'\nThe 5 coefficients with the most weight are: {weight_coefficients}')

data_point = np.array([6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4])
data_point_scaled = scaler.transform(data_point.reshape(1, -1))
pred = reg.predict(data_point_scaled)
print(f'The predicted mpg for the data point is: {pred[0]:.2f}')

#Set marker properties
markersize = df[weight_coefficients[3]]
markercolor = df['mpg']
markershape = df[weight_coefficients[4]].replace(0, 'square').replace(1, 'circle')

#Make Plotly figure
fig1 = go.Scatter3d(x = df[weight_coefficients[0]],
                    y = df[weight_coefficients[1]],
                    z = df[weight_coefficients[2]],
                    marker = dict(size = markersize,
                                color = markercolor,
                                symbol = markershape,
                                opacity = 0.9,
                                reversescale = True,
                                colorscale = 'Blues'),
                    line = dict (width = 0.02),
                    mode = 'markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene = dict(xaxis = dict( title = 'wt'),
                                yaxis = dict( title = 'disp'),
                                zaxis = dict(title = 'hp')),)

#Plot and save html
plotly.offline.plot({'data': [fig1],
                     'layout': mylayout},
                     auto_open = True,
                     filename = ('6DPlot.html'))
