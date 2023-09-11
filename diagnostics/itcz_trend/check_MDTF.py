# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:15:51 2023

@author: JillianW
"""
import xarray as xr
ev = xr.open_dataset('C:/ubuntu/ITCZ/CMIP6_his_ev_pr/ACCESS-CM2.ev.nc')['evspsbl']
 os.environ['lon_var'] = 'lon'