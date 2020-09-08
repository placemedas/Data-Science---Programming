# Please ensure weather_model(unzipped), stat.csv ,global_pred.csv and test_pred.csv
# files are created using weather_spark.py before running this program

import elevation_grid as eg
import sys
import geopandas as gd
import descartes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from numpy import array
from mpl_toolkits.basemap import Basemap


# Function to visualize the results of max temperature distributions over two different periods
def scatter_plot(stat):
    # Plot configurations
    figsize = (16, 10)
    plt.rc('legend', fontsize=10)
    plt.rcParams['legend.title_fontsize'] = 'large'

    # Geometry calculation of co-ordinates in dataset
    gdf1 = gd.GeoDataFrame(stat, geometry=gd.points_from_xy(stat.longitude, stat.latitude))
    # Obtaining the world map
    world = gd.read_file(gd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='gray', edgecolor='black', figsize=figsize)
    # Plotting the data
    gdf1.plot(ax=ax, column='category', categorical=True, legend=True, markersize=15, cmap='gist_rainbow',
              figsize=figsize, legend_kwds={'loc': 'lower left'})
    # Title and legend setups
    leg = ax.get_legend()
    leg.set_title("Change in celsius")
    plt.title('Temperature Change between 1977-''96 & 1997 - 2017', fontsize=30)
    plt.show()


def denseplot(result_df1):
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.title("Dense Plot of Predicted Temperatures on 20 January,2020", fontsize=20)

    # Creating basemap to make this plot
    mp = Basemap(projection='robin', lon_0=0, resolution='c')
    mp.drawcoastlines()
    mp.fillcontinents(color='green', lake_color='aqua')
    # draw parallels and meridians.
    mp.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0], fontsize=10)
    mp.drawmeridians(np.arange(0., 360., 60.), labels=[0, 0, 0, 1], fontsize=10)
    # draw map buondaries
    mp.drawmapboundary(fill_color='aqua', linewidth=2)
    # Plotting of predictions across map
    lats, lons = np.meshgrid(np.arange(-90, 90, .5), np.arange(-180, 180, .5))
    z = result_df1["prediction"].values.reshape(lons.shape)
    mp.pcolormesh(lons, lats, z, cmap='rainbow', zorder=2, alpha=0.35, linewidth=1, latlon=True)
    cbar = mp.colorbar()
    plt.show()

def scatter_plot_error(t_pred):
    #Plotting Configurations
    figsize = (16, 10)
    plt.rc('legend', fontsize=10)
    plt.rcParams['legend.title_fontsize'] = 'large'
    #Geometry calculation of co-ordinates in dataset
    gdf1 = gd.GeoDataFrame(t_pred,geometry=gd.points_from_xy(t_pred.longitude, t_pred.latitude))
    #Setting the world map
    world = gd.read_file(gd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='gray',edgecolor='black',figsize=figsize)
    #Plotting of data
    gdf1.plot(ax=ax,column='category', categorical=True,legend=True,markersize=20,cmap='gist_rainbow',figsize=figsize,legend_kwds={'loc': 'lower left'})
    # Legend and Title settings
    leg = ax.get_legend()
    leg.set_title("Regression Error")
    plt.title('Regression Error of Predictions for 2015',fontsize=30)
    plt.show()


def main():

    # Part 2 - Task a - Visualization of maximum daily distribution
    stat = pd.read_csv("stat.csv")
    #print(stat)
    scatter_plot(stat)

    #Part 2 - Task b1 - Visualization of Heat Map on Temperature predictions
    result_df1 = pd.read_csv("global_pred.csv")
    denseplot(result_df1)

    #Part 2 - Task b2 - Visualization of Regression Error on Temperature predictions
    t_pred = pd.read_csv("test_pred.csv")
    scatter_plot_error(t_pred)

if __name__ == '__main__':
    main()






