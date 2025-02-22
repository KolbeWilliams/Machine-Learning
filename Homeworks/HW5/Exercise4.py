#Exercise 4:
import pandas as pd
import plotly
import plotly.graph_objs as go

#Read materials data from csv
data = pd.read_csv("materials.csv")

#Set marker properties
markercolor = data['Temperature']

#Make Plotly figure
fig1 = go.Scatter3d(x = data['Time'],
                    y = data['Pressure'],
                    z = data['Strength'],
                    marker=dict(color = markercolor,
                                opacity= 1,
                                reversescale = True,
                                colorscale = 'Blues',
                                size = 5),
                    line = dict (width = 0.02),
                    mode = 'markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene = dict(xaxis = dict( title = 'Time'),
                                yaxis = dict( title = 'Pressure'),
                                zaxis = dict(title = 'Strength')),)

#Plot and save html
plotly.offline.plot({'data': [fig1],
                     'layout': mylayout},
                     auto_open = True,
                     filename = ('4DPlot.html'))
