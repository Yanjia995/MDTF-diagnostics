# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:39:18 2023

@author: JillianW
"""
import os
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,linregress
print('\n=======================================')
print('BEGIN ITCZ_trend.py ')
print('=======================================\n')
### 1) Loading model data files: ##########
pr_mod_path = os.environ["PR_FILE"]# load precipitation netcdf data
pr_var_name = os.environ["pr_var"]
pr_time_coord_name = os.environ["time_coord"]
evp_mod_path = os.environ["EVSPSBL_FILE"]# load evaporarion netcdf data
evp_var_name = os.environ["evspsbl_var"]
evp_time_coord_name = os.environ["time_coord"]
rlut_mod_path = os.environ["RLUT_FILE"]# load TOA longwave radiation netcdf data
rlut_var_name = os.environ["rlut_var"]
rlut_time_coord_name = os.environ["time_coord"]
w_mod_path = os.environ["WAP500_FILE"]# load TOA longwave radiation netcdf data
w_var_name = os.environ["wap500_var"]
w_time_coord_name = os.environ["time_coord"]
### 2) Doing calculations: ##########
 # get only field of ITCZ
pr_mod_ITCZ = xr.open_dataset(pr_mod_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = pr_time_coord_name)*86400
evp_mod_ITCZ = xr.open_dataset(evp_mod_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = evp_time_coord_name)*86400
EP_mod_ITCZ = evp_mod_ITCZ - pr_mod_ITCZ
rlut_mod_ITCZ = xr.open_dataset(rlut_mod_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = rlut_time_coord_name)
w_mod_ITCZ = xr.open_dataset(w_mod_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = w_time_coord_name)
# crop model data from year
start_year = np.int(os.getenv("start_year"))  
end_year = np.int(os.getenv("end_year"))
EP_mod = EP_mod_ITCZ.sel(year = slice(start_year, end_year))
rlut_mod = rlut_mod_ITCZ.sel(year = slice(start_year, end_year))
w_mod = w_mod_ITCZ.sel(year = slice(start_year, end_year))
ITCZ_ep_mod = np.ones(35, dtype='float')
ITCZ_rlut_mod = np.ones(35, dtype='float')
ITCZ_w_mod = np.ones(35, dtype='float')
for i in range(35):
    ITCZ_ep_mod[i] = np.sum(EP_mod[i,:,:] < 0.2)/EP_mod[i,:,:].size*100
    ITCZ_rlut_mod[i] = np.sum(rlut_mod[i,:,:] < 250)/rlut_mod[i,:,:].size*100
    ITCZ_w_mod[i] = np.sum(w_mod[i,:,:] < 0)/w_mod[i,:,:].size*100
### 3) Load observed data and calcualting: ########## 
obs_dir = "{OBS_DATA}/".format(**os.environ)
pr_ob_path = obs_dir + 'gpcp.pr.1980-2014.nc.nc'
evp_ob_path = obs_dir + 'era.ev.1980-2014.lon.nc'
rlut_ob_path = obs_dir + 'noaa.olr.1975-2022.nc'
w_ob_path = obs_dir + 'w500.1960-2020.nc'
pr_ob_ITCZ = xr.open_dataset(pr_ob_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = pr_time_coord_name)*86400
evp_ob_ITCZ = xr.open_dataset(evp_ob_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = evp_time_coord_name)*86400
EP_ob_ITCZ = evp_ob_ITCZ - pr_ob_ITCZ
rlut_ob_ITCZ = xr.open_dataset(rlut_ob_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = rlut_time_coord_name)
w_ob_ITCZ = xr.open_dataset(w_ob_path).sel(lat = slice(30,-30)).groupby('time.year').mean(dim = w_time_coord_name)
# crop obel data from year
start_year = np.int(os.getenv("start_year"))  
end_year = np.int(os.getenv("end_year"))
EP_ob = EP_ob_ITCZ.sel(year = slice(start_year, end_year))
rlut_ob = rlut_ob_ITCZ.sel(year = slice(start_year, end_year))
w_ob = w_ob_ITCZ.sel(year = slice(start_year, end_year))
ITCZ_ep_ob = np.ones(35, dtype='float')
ITCZ_rlut_ob = np.ones(35, dtype='float')
ITCZ_w_ob = np.ones(35, dtype='float')
for i in range(35):
    ITCZ_ep_ob[i] = np.sum(EP_ob[i,:,:] < 0.2)/EP_ob[i,:,:].size*100
    ITCZ_rlut_ob[i] = np.sum(rlut_ob[i,:,:] < 250)/rlut_ob[i,:,:].size*100
    ITCZ_w_ob[i] = np.sum(w_ob[i,:,:] < 0)/w_ob[i,:,:].size*100
### 4) Visualizing and Saving plots: ##########
def plot_and_save_figure(model_data, ob_data, variable_name, title_string):
    # linear regression
    model_m,model_b = linregress(model_data.year.values, model_data.values)
    ob_m,ob_b = linregress(ob_data.year.values, ob_data.values)
    # initialize the plot
    fig, ax = plt.subplots()
    ax.set_xlabel('year')
    ax.set_ylabel('Area Coverage %')
    ax.set_ylim([ 36,52])
    ax.set_yticks([  40, 45, 50])
    color = 'tab:red'
    ax.plot(model_data.year, model_data, color=color, 
             label = 'OLR ' + format(model_m*10, '0.2f') + '% decade$^{-1}$')
    ax.plot(model_data.year, model_m*model_data.year+model_b, 
             color=color, linestyle = '--')
    color = 'black'
    ax.plot(ob_data.year, ob_data, color=color, 
            label = 'OLR ' + format(ob_m*10, '0.2f') + '% decade$^{-1}$')
    ax.plot(ob_data.year, ob_m*model_data.year+ob_b, 
            color=color, linestyle = '--')
    # save the plot in the right location
    plot_path = "{WK_DIR}/{model_or_obs}/PS/{model_or_obs}_trend.eps".format(**os.environ)
    plt.savefig(plot_path, bbox_inches='tight')
# end of function

# set an informative title using info about the analysis set in env vars
title_string = "{CASENAME}: ITCZ area coverage bsaed on E-P ({FIRSTYR}-{LASTYR})".format(**os.environ)
# Plot the model data:
plot_and_save_figure(ITCZ_ep_mod, ITCZ_ep_ob, 'E-P', title_string)
title_string = "{CASENAME}: ITCZ area coverage bsaed on {tas_var} ({FIRSTYR}-{LASTYR})".format(**os.environ)
plot_and_save_figure("model", title_string, model_mean_tas)
