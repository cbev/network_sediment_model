# -*- coding: utf-8 -*-
"""
Network sediment model for Elwha watershed

@author: Claire Beveridge, University of Washington (cbve@uw.edu)
"""
#%% Processing Options-- 'yes' or 'no'
setup_mg='no'
process_met='no'
plot_met='no'
process_obs='no'
process_qmod='no'
plot_q='no'
process_network='yes' # keep this as 'yes' unless there is a deliberate reason not to
recycle_run='yes' 
recycle_run_directory='C:/Users/Claire/Documents/GitHub/cbev_projects/elwha_network_model/output/runs/dhsvm_streamflow/2018-10-10_1000'
start_date='1/1/1915' #1/1/1915

# Import modules and define functions
import os
import csv
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import pickle
import scipy.stats

# Landlab modules
from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.io import read_esri_ascii
#%% Parameters
# Constants
g=9.81                       # acceleration of gravity (m/s2)
rho_w=1000                  # density of water (kg/m3)
v=10**-6                    # kinematic viscosity of water (m2/s)
sg=2.65                     # specific gravity 
bd_fine=1.13                # bulk density of fine sediments in reservoir 
bd_coarse=1.71              # bulk density of coarse sediments in reservoir 
grid_cell_m=30              # length of grid cell

# Names (for figures, print outs etc.)
dem_name='Elwha Watershed (30m DEM)'
loc_name='Lake Mills Tributary'
drainage_area_gc=681429731  # m2- drainage area of Glines Canyon Dam
drainage_area_mills=512954011 #m2- drainage area of Lake Mills gage

# Directories
## sediment model
model_dir='C:/Users/Claire/Documents/GitHub/cbev_projects/elwha_network_model/'
model_folder=model_dir+'current_version/'
model_input_folder=model_dir+'input/'
model_data_folder=model_dir+'data/dhsvm_streamflow/output_32/'
model_output_folder=model_dir+'output/runs/dhsvm_streamflow/'
model_obs_folder=model_dir+'observations/'
## GIS and DHSVM
gis_folder='D:/GoogleDrive/Elwha/GIS/'
dhsvm_dir='D:/GoogleDrive/Elwha/Elwha_DHSVM/'
dhsvm_forcing_dir=dhsvm_dir+'forcings/'
dhsvm_gis_dir=dhsvm_dir+'forcings/GIS/'
dhsvm_forcing_folder=dhsvm_forcing_dir+'data/WRFadj/'
dhsvm_network_folder=dhsvm_dir+'network/1km2_0.25-2.5/'

# Files
## model_input_folder files
dem_input='elwhaws30demf.txt' # full Elwha watershed- filled
mask_input='elwhaws30mask_1km2.txt' # dhsvm stream watershed masks
res_input='res_mask.txt' # reservoir mask
slp_input='elwhaws30slp.txt' # slope of watershed
ca_input='elwhaws30ca.txt' # contributnig area of watershed
network_input='streamfile_1km2_0.25-2.5.txt'

## model_obs_folder
streamflow_obs_input_xls='Mills_USGS-Daily_Q-H-T-SS_1994-2015.xlsx'
streamflow_obs_input_txt='Mills_USGS-Daily_Q-H-T-SS_1994-2015.txt'

## dhsvm_dir
mappingfile='ElwhaClimatePoints_UTM_Elev.csv' # csv file with lat, long and elevation. Header should include 'LAT','LONG_', and 'ELEV'

## dhsvm_network_folder files
networkdat_input='stream.network.dat'
map_input='stream.map.dat'

## dhsvm_forcing_folder shapefiles
ws_bnd_file=dhsvm_gis_dir+'elwha_ws_bnd_wgs84.shp'
met_grid_file=dhsvm_gis_dir+'gc_met_sta.shp'

## dhsvm_output_folder
streamflow_input='Streamflow.Only'

# Parameterization
## time parameters
forcing_start_date=datetime.date(1915,1,1)
forcing_end_date=datetime.date(2011,12,31)

q_mills_start_1=datetime.date(1994,3,26) # al streamflow data
q_mills_end_1=datetime.date(1998,5,31)
q_mills_start_2=datetime.date(2004,2,18)
q_mills_end_2=datetime.date(2011,9,30)
q_mills_start_wy_1=datetime.date(1994,10,1) # water years only
q_mills_end_wy_1=datetime.date(1997,9,30)
q_mills_start_wy_2=datetime.date(2004,10,1)
q_mills_end_wy_2=datetime.date(2011,9,30)

## hydraulic geometry (HG) parameters (note: a is constant, b is exponent)
ref_stream= 144 # DHSVM stream number where observations are collected that is used to scale other streams. For Elwha, this is the Lake Mills stream.
q_mult=1

# stream width @ ref stream
w_const_ref=41.6

## stream depth HG parameters @ ref stream
a_d_ref=0.24
b_d_ref=0.41

### stream velocity HG parameters @ ref stream
a_u_ref=0.10
b_u_ref=0.58

## stream width HG upstream exponent
exp_w_us=0.5

## stream depth HG upstream exponent
exp_d_us=0.4

## stream velocity HG upstream exponent
exp_u_us=0.1

# roughness parameterization
a_n=1.08
b_n=-0.44

## sediment transport parameterization per Wilcock and Crowe, 2003
# Channel grain size properties at Lake mills gage stream
Ch_Fs_LM=0.37
d90ch_LM=0.0275 # m -  d90 of channel bed= 27.5 mm=0.0275 m
dsandch_LM=0.00093 # m- d_sand of channel bed= 0.93 mm = 0.00093 m
dgravelch_LM=0.0132 # m- d_gravel of channel bed= 13.2 mm = 0.0132 m
dmeanch_LM=0.0125 # m- d_mean of channel bed= 12.5 mm = 0.0125 m

# Wilcock- Crowe Equation Parameters
A_gravel=14
chi_gravel=0.894
exp_gravel=0.5
phi_prime_gravel=1.35

A_sand=14
chi_sand=0.894
exp_sand=0.5
phi_prime_sand=1.35

# Suspended Sediment Equation Parameters
# Rating Curve at Lake Mills gage
a_s=1.17*10**-4 # regression coefficient from Curran et al., 2009
b_s_c=3        # regression coefficient for SS concentration from Curran et al., 2009
b_s_l=4        # regression coefficient for SS load from Curran et al., 2009
cf_s=1.07      # log-regression correction factor from Curran et al., 2009
K_ss=0.0864       # unit conversion factor from Curran et al., 2009  

# Reservoir Trap Efficiency for Suspended Sediment
rte=0.86

# Network suspended sediment parametrization
# Key references: Patil et al., 2012
# for suspended sand depostiion
d_sand_ss=0.00025 # 0.0004m =0.4mm=400 um
d_star_ss=(((sg-1)*g*d_sand_ss**3)/(v**2))**(1/3) # unitless grain size 
v_ss=(v/d_sand_ss)*(np.sqrt((1/4)*(24/1.5)**(2/1)+((4*d_star_ss**3)/(3*1.5))**(1/1))-(1/2)*(24/1.5)**(1/1))**1 # falling velocity of sand (m/s)              

c0_ss=1.1038
c1_ss=2.6626
c2_ss=5.6497
c3_ss=0.3822
c4_ss=-0.6174
c5_ss=0.1315
c6_ss=-0.0091
A_ss=1.3*10**-7

# for bedload sand equations
pcnt_sb_er=0.08  # percent of sand on the bed surface that can be eroded into suspended sediment
d_sand_sb=0.00025  #dsandch_LM
d_star_sb=(((sg-1)*g*d_sand_sb**3)/(v**2))**(1/3) # unitless grain size 
v_sb=(v/d_sand_sb)*(np.sqrt((1/4)*(24/1.5)**(2/1)+((4*d_star_sb**3)/(3*1.5))**(1/1))-(1/2)*(24/1.5)**(1/1))**1 # falling velocity of sand (m/s)              

Re_p=v_sb*dsandch_LM/v
if Re_p>2.36:
    alp_1=1
    alp_2=0.6
else:
    alp_1=0.586
    alp_2=1.23

# for silt equations
pcnt_m_er=0.20   # percent of silt on the bed surface that can be eroded into suspended sediment
tau_c_fines=0.015*(bd_fine*rho_w-rho_w)**0.73
a_w_m=0.08
n_w_m=1.65
b_w_m=3.5
m_w_m=1.88
c1_m=0.15
c2_m=b_w_m/((2*m_w_m-1)**0.5)

## geomorphology
Drate=0.22/365.25 # denudation rate [mm/day]- based on computations/literature
beta_mw=20*365.25 # lag time [days] between mass wasting events- exponenetial distribution parameter
pct_ss_mw=0.1
pct_m_mw=0.1

mw_composition= 'set' # 'stochastic' or 'set'
pct_g_mw_dep=0.333 #0.333
pct_s_mw_dep=0.333 #0.366
pct_m_mw_dep=0.333 #0.300

init_depth=0.5 # Set initial depth of sediment in channels [m] 

abrasion_alpha=0.027
bed_alpha=0.02

# Other
bypass_threshold=90 # Min length a segment must be, ohterwise sediment will transport directly through. This is to mitigate runaway accumulation

#%% Functions
def read_in_longlats(mappingfile):
    maptable=[]
    with open(mappingfile, 'r') as csvfile:
        longlat = csv.reader(csvfile, delimiter=',')
        for row in longlat:
            maptable.append(row)
    csvfile.close()
    return(maptable)

#  "Find nearest" function
def find_nearest(array,value):
    val = (np.abs(array-value)).argmin()
    return array[val]

# Upload observed data
def create_q_obs_df(file_name, drainage_area):
    q=pd.read_excel(file_name, sheetname='data', skiprows=[0], header=None, usecols='A:D')
    q.columns=['year','month','day','flow_cfs']
    q_dates=pd.to_datetime(q.loc[:,['year','month','day']])
    q.set_index(q_dates, inplace=True)
    q.drop(['year','month','day'],axis=1, inplace=True)
    q_cms=q.flow_cfs/(3.28084**3)
    q_mmday=q_cms*1000*3600*24/drainage_area
    q=pd.concat([q_cms, q, q_mmday],axis=1)
    q.columns=['flow_cms','flow_cfs', 'flow_mmday']
    return q

def import_obs_folders(obs_folder,streamflow_obs_input_txt):
    os.chdir(obs_folder)
    LM_usgs= np.genfromtxt(streamflow_obs_input_txt,skip_header=1,dtype=str)                   
    # Extract Dates- get year, month, day into datetime objects:
    n_1=len(LM_usgs[:,0]) # n is number of days in the record
    date_LM=np.full(n_1,'', dtype=object) # Preallocate date_1 matrix
    for x in range(0,n_1): # Cycle through all days of the year
        date_LM_temp=datetime.date(int(LM_usgs[x,0]),int(LM_usgs[x,1]),
                                               int(LM_usgs[x,2]))
        # make numpy array of individual temporary datetime objects
        date_LM[x]=date_LM_temp # enter temporary object into preallocated date matrix
    del(date_LM_temp) # delete temporay object
    
    # Extract remaining variables and convert to standard units:
    Q_day=np.array((LM_usgs[:,3]), dtype='float64')
    Q_day=Q_day/(3.28084**3) # convert from ft^3/s to m^3/s
    SSC_day=np.array(LM_usgs[:,7], dtype='float64')
    SSL_day=np.array(LM_usgs[:,9], dtype='float64')
    T_day=np.array(LM_usgs[:,11], dtype='float64')
    
    LM_data=pd.DataFrame({"Q_m3s": Q_day, "SSC_mgL": SSC_day,
                          "SSL_tonsday":SSL_day, "T_fnu":T_day},index=pd.to_datetime(date_LM))
    
    # Now find the 50 % exceedance flow at Lakem Mills gage- use complete water years only
    Q_curve=LM_data.loc[datetime.date(1994, 10, 1):datetime.date(1997, 9, 30), 'Q_m3s'].values
    Q_curve=np.append(Q_curve,LM_data.loc[datetime.date(2004, 10, 1):datetime.date(2011, 9, 30), 'Q_m3s'].values)
    
    # FUTURE:  Use calibrated DHSVM outputs instead?
    Q_RC=np.sort(Q_curve)
    n_curve=len(Q_curve)
    ep_Q_curve=np.array(range(1,n_curve+1))/(n_curve+1)
    cum_pcntl=ep_Q_curve
    Q_LM_50EP=Q_RC[np.where(cum_pcntl==find_nearest(cum_pcntl, 0.50))] # 50th percentile
            
    return (LM_data, Q_LM_50EP)

def compute_NSE_rs (modeled, observed):
    NSE=1-((np.sum((modeled-observed)**2))/(np.sum((modeled-np.mean(observed))**2)))
    WC,WC_0,WC_r, WC_p, WC_err=scipy.stats.linregress(modeled, observed)
    r2=WC_r**2
    print('r2=',r2)
    print('NSE=',NSE)
    return NSE, r2

def setup_landlab_mg(input_folder):
    os.chdir(input_folder)  
    #Import the DEM grid (exported from GIS)
    (mg0, z0) = read_esri_ascii(dem_input, name='topographic__elevation') # import DEM (units of m)   
    #Now add additional grids (exported from GIS), as needed, to the same grid, mg0. Note that for Elwha we need to start with mg0 because then we decrease the size:
    (mg0, slp0) = read_esri_ascii(slp_input, name='slope', grid=mg0) # add slope raster field (unitless slope) (derived from TauDEM, d-infinity algorithm)
    (mg0, ca0) = read_esri_ascii(ca_input, name='cont_area', grid=mg0) # add contributing area (square meters) (derived from TauDEM, d-infinity algorithm)
    (mg0, trib_mask0) = read_esri_ascii(mask_input, name='trib_mask', grid=mg0) # add mask for contributing area
    (mg0, res_mask0) = read_esri_ascii(res_input, name='res_mask', grid=mg0) # add mask for reservoirs
    
    #Print the keys in mg to see what was uploaded 
    print('Model grid imported. Model grid keys:', mg0.at_node.keys())
    return (mg0, z0, slp0, ca0, trib_mask0, res_mask0)

def setup_network(input_folder):                      
    os.chdir(model_input_folder)  
    network=pd.read_table(network_input, delimiter='\t', index_col=0,\
                          usecols=[0, 5, 6, 7, 13, 14, 15])
    network.columns=['segment_length_m','local_ca','dest_channel_id',\
                     'segment_slope','total_ca_mean','segment_order']
    network['local_ca']=network['local_ca']*(grid_cell_m**2) # change from number of cells to area
    
    return (network)
      

def compute_channel_properties(ref_stream, a_u_ref, b_u_ref, a_d_ref, b_d_ref, a_n, b_n, ng_obs_bar, S, total_ca_ref, total_ca, width, Qref, Q):  
    rho_w=1000
    g=9.81
    # ref stream values for given flow
    Dref=a_d_ref*Qref**b_d_ref
    #Uref=a_u_ref*Qref**b_u_ref
    # stream-of-interest values for given flow
    D=Dref*(total_ca**exp_d_us)/(total_ca_ref**exp_d_us)
    #U=Uref*(total_ca**exp_u_us)/(total_ca_ref**exp_u_us)
    U=Q/(D*width)
    ng=ng_obs_bar*a_n*D**b_n
    tau=rho_w*g*((ng*U)**(3/2))*S**(1/4)
    u_star=(tau/rho_w)**0.5
    return tau, u_star         
        
def run_wc2003_2F_model (tau, tau_r_sand,tau_r_gravel):
    # Constants
    A=14
    chi=0.894
    exp=0.5
    phi_prime=1.35 
             
    # Run WC 2003 Two-Fraction Model
    phi_gravel=tau/tau_r_gravel
    phi_sand=tau/tau_r_sand
    
    if phi_gravel<phi_prime:
        Wstar_gravel=0.002*(phi_gravel)**7.5
    elif chi/((phi_gravel)**exp)>=1: # Checka that term in paraentheses is not negative
       Wstar_gravel=0
    else:
        Wstar_gravel=A*((1-(chi/((phi_gravel)**exp)))**(4.5))

    if phi_sand<phi_prime:
        Wstar_sand=0.002*(phi_sand)**7.5
    elif chi/((phi_sand)**exp)>=1: # Checka that term in paraentheses is not negative
       Wstar_sand=0
    else:
        Wstar_sand=A*((1-(chi/((phi_sand)**exp)))**(4.5))        
    return (Wstar_gravel, Wstar_sand)    

#%%
# Setup Landlab model grid (mg)
os.chdir(model_folder)
if setup_mg=='yes':
    os.chdir(model_folder)
    [mg0, z0, slp0, ca0, trib_mask0, res_mask0]=setup_landlab_mg(model_input_folder)
    # Set no data nodes to closed
    mg0.set_nodata_nodes_to_closed(mg0.at_node['topographic__elevation'], -9999.)
    
    # Store variables relevant to your grid
    m0=mg0.number_of_node_rows # m is number of rows
    n0=mg0.number_of_node_columns # n is number of columns
    
    # Extract a subset of area to conduct analysis and create a new model grid
    # For Elwha, this is upstream of Glines Canyon Dam
    # INPUT desired x extent and y extent of mg
    new_x_extent=[0, n0*30] # 1000, 3400
    new_y_extent=[2500, 45000] # 500, 5000; 
    
    # Extract values of interest from mg0 grid
    rows=[int(np.round(new_y_extent[0]/30.)),int(np.round(new_y_extent[1]/30.))] # rows to extract from mg0
    cols=[int(np.round(new_x_extent[0]/30.)),int(np.round(new_x_extent[1]/30.))] # columns to extract from mg0
    
    start_row=rows[0]
    end_row=rows[1]
    start_col=cols[0]
    end_col=cols[1]
    
    ncols=end_col-start_col # number of rows in new grid
    nrows=end_row-start_row # number of columns in new grid
    
    new_grid_size=[nrows*ncols]
    
    ind=np.empty(new_grid_size)
    z=np.zeros(new_grid_size)
    slp=np.empty(new_grid_size)
    ca=np.empty(new_grid_size)
    trib_mask=np.empty(new_grid_size)
    res_mask=np.empty(new_grid_size)
    
    k=0 
    for i in range (0,nrows):
        for j in range (0,ncols):
            ind[k]=n0*(start_row+i)+start_col+j
            z[k]=z0[int(ind[k])]
            slp[k]=slp0[int(ind[k])]
            ca[k]=ca0[int(ind[k])]
            trib_mask[k]=trib_mask0[int(ind[k])]
            res_mask[k]=res_mask0[int(ind[k])]
            k=k+1
    
    # Create mg for area of interest for the analysis
    mg=RasterModelGrid((nrows,ncols), spacing=(30.,30.))
    mg.add_field('node','topographic__elevation', z)
    mg.add_field('node','slope', slp)
    mg.add_field('node','cont_area', ca)
    mg.add_field('node','trib_mask', trib_mask)
    mg.add_field('node','res_mask', res_mask)
    
    # Set core nodes of Lake Mills watershed by closing boundary and then assigning core nodes
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    mg.set_nodata_nodes_to_closed(mg['node']['trib_mask'], -9999.)
    mg.set_nodata_nodes_to_closed(mg['node']['topographic__elevation'], -9999.)
    mg.set_nodata_nodes_to_closed(mg['node']['slope'], -9999.)
    core_nodes_ws=mg.core_nodes
    
    # Set watershed (upstream of Lake Mills) boundary conditions
    # Set the Lake Mills reservoir as the outlet
    res_nodes=np.where(mg.at_node['res_mask']==1)[0] # find nodes of the reservoir
    mg.set_watershed_boundary_condition_outlet_id(res_nodes, z, nodata_value=-9999.) # Set the outlet ID
    
    # Close boundarues at grid edges 
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True) # Set status of nodes along the specified sides of a raster grid as closed
    # Plot grids
    #imshow_grid(mg, 'trib_mask', cmap='inferno')
    imshow_grid(mg, 'topographic__elevation', cmap='inferno')

# Import and plot gridded met stations
if process_met=='yes':
    # Import gridded climate mapping file with the lat, long, and elev of each point
    os.chdir(dhsvm_forcing_dir)
    maptable = read_in_longlats(mappingfile) # maptable is a list of the rows/columns
    # from the mapping file. The top row is the header of each column. 
    station=maptable[0].index('FID')
    latitude=maptable[0].index('LAT')
    longitude=maptable[0].index('LONG')
    elevation=maptable[0].index('ELEV')
    
    station_list=[]
    lat_list=[]
    long_list=[]
    elev_list=[]
    
    for row in maptable:
        if maptable.index(row)!=0:
            station_list.append(row[station])
            lat_list.append(row[latitude])
            long_list.append(row[longitude])
            elev_list.append(int(row[elevation]))
    
    # Create a data frame and geodataframe with station number, lat, long, and elevation 
    met_df=pd.DataFrame({"station": station_list,"latitude": lat_list,
                         "longitude": long_list,"elevation": elev_list})
    met_geometry=[Point(xy) for xy in zip(pd.to_numeric(met_df['longitude']),
                  pd.to_numeric(met_df['latitude']))]
    met_gdf=gpd.GeoDataFrame(met_df,geometry=met_geometry)
    n_stations=len(met_df) # number of stations
    
    # Upload forcing cell geometry (relative areas of grid cells for area of interest)
    station_geom=pd.read_excel('met_forcing_gis.xlsx', sheetname='glines')
    station_geom.set_index('Cell_ID', inplace=True)
    
    #Create xarray of station data
    station_area_weights_ds=xr.DataArray(station_geom.Area_m2/drainage_area_gc, 
                                               coords=[station_geom.index], dims=['station'])
    
    # Import WRF 2016 forcing data and put in data frame
    os.chdir(dhsvm_forcing_folder)
    # compile names of met data files
    file_names=[]
    for i in range(0, n_stations):
        file_names.append('_'.join(['WRFadj',met_df.latitude[i],met_df.longitude[i]]))
    
    # compile list of focing data frame for each station
    wrf2016_dates=pd.date_range(forcing_start_date,forcing_end_date)
    wrf2016=[]
    for i in range(0, n_stations):
        wrf2016.append(pd.read_table(file_names[i], header=None))
        wrf2016[i].columns=['precip','tmax', 'tmin', 'wind']
        wrf2016[i].set_index(wrf2016_dates, inplace=True) # set the index as the date
    os.chdir(model_data_folder)
    pickle.dump(met_df, open("met_df.py", "wb"))
    pickle.dump(met_gdf, open("met_gdf.py", "wb"))
    pickle.dump(station_area_weights_ds, open("station_area_weights_ds.py", "wb"))
    pickle.dump(wrf2016, open("wrf2016.py", "wb"))

# Plot gridded climate points
if plot_met=='yes':
    ws_bnd=gpd.read_file(ws_bnd_file)
    met_grid_poly=gpd.read_file(met_grid_file)
    f, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_title('Gridded Meteorology Points: Elwha Watershed')
    ws_bnd.plot(ax=ax, cmap='Set3')
    met_gdf.plot(ax=ax, marker='^', color='black', edgecolor='white')
    met_grid_poly.plot(ax=ax, color='none', edgecolor='black')
    ax.set_xlim([-123.7, -123.1])
    ax.set_ylim([47.6, 48.2])
    plt.axis('equal')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

# Import observed gage data
if process_obs=='yes':
    os.chdir(model_input_folder)
    LM_data, LM_Qobs_50EP=import_obs_folders(model_obs_folder,streamflow_obs_input_txt)
    qobs_mills=create_q_obs_df(streamflow_obs_input_xls, drainage_area_mills)
    os.chdir(model_data_folder)
    pickle.dump(qobs_mills, open("qobs_mills.py", "wb"))
    pickle.dump(LM_Qobs_50EP, open("LM_Qobs_50EP.py", "wb"))
    pickle.dump(LM_data, open("LM_data.py", "wb"))
#
if process_qmod=='yes':
    os.chdir(dhsvm_output_folder)
    stream_columns=['Date', 101,103,104,105,106,110,111,112,115,116,117,118,\
     120, 121,122,123,124, 127,128,129,130,132,133,134,135,136,137,138,139,\
     140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,\
     158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,\
     176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,\
     194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,\
     212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,\
     230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,\
     248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,\
     266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,\
     284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,\
     302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,\
     320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,\
     338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,\
     356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,\
     374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,\
     392,393,394,395,396,397,398,399,400,401,402,403]
    
    forcing_date_range=pd.date_range(forcing_start_date, forcing_end_date)
    streamflow=pd.DataFrame(index=forcing_date_range, columns=stream_columns)
    chunksize = 8
    i=0
    for chunk in pd.read_table(streamflow_input, skiprows=[0,1], header=None,\
                               sep='\s+', chunksize=chunksize):
        chunk.columns=stream_columns
        streamflow.loc[forcing_date_range[i],:]=chunk.sum(axis=0)[1::]/86400
        i=i+1
    last_day=pd.read_table(streamflow_input, skiprows=np.arange(0,(i-1)*8+2), 
                           header=None,sep='\s+')
    last_day.columns=stream_columns
    streamflow.loc[forcing_date_range[i],:]=last_day.sum(axis=0)[1::]/86400
    Qmod_median=streamflow.median(axis=0)
    os.chdir(model_data_folder)
    pickle.dump(streamflow, open("streamflow.py", "wb"))
    pickle.dump(Qmod_median, open("Qmod_median.py", "wb"))

# Plot modeled versus observed streamflow
if plot_q=='yes':
    if 'streamflow' not in locals():
        os.chdir(model_data_folder)
        streamflow=pickle.load(open('streamflow.py', 'rb'))
    if 'qobs_mills' not in locals():
        os.chdir(model_data_folder)
        qobs_mills=pickle.load(open('qobs_mills.py', 'rb'))
    qmod_mills=streamflow[ref_stream]
    # Time Series
    fig1, (ax1, ax2)=plt.subplots(2,1, figsize=(8,4))
    ax1.plot(qobs_mills.flow_cms[q_mills_start_1:q_mills_end_1].index, 
             qobs_mills.flow_cms[q_mills_start_1:q_mills_end_1].values, 
             'k',linewidth=1, label='Observed')
    ax1.plot(qmod_mills[q_mills_start_1:q_mills_end_1].index, 
             q_mult*qmod_mills[q_mills_start_1:q_mills_end_1].values, 
             'r',linewidth=1, label='Modeled')
    #plt.xlabel('Date')
    ax1.set_ylabel('Streamflow (cms)')
    ax1.legend(loc='best')
    ax1.set_title('Observed vs Modeled Daily Streamflow at Lake Mills gage\n3/27/1994-5/31/1998')
    
    #plt.subplots(2,1, figsize=(8,4))
    ax2.plot(qobs_mills.flow_cms[q_mills_start_2:q_mills_end_2].index, 
             qobs_mills.flow_cms[q_mills_start_2:q_mills_end_2].values, 
             'k',linewidth=1, label='Observed')
    ax2.plot(qmod_mills[q_mills_start_2:q_mills_end_2].index, 
             q_mult*qmod_mills[q_mills_start_2:q_mills_end_2].values, 
             'r',linewidth=1, label='Modeled')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Streamflow (cms)')
    ax2.set_title('2/18/2004-9/30/2011')
    
    # One-to-One Plot
    fig2, ax1=plt.subplots(1,1, figsize=(6,6))
    plt.loglog(q_mult*qmod_mills[q_mills_start_1:q_mills_end_1].values,
               qobs_mills.flow_cms[q_mills_start_1:q_mills_end_1].values,
               'bo',markersize=1)
    plt.loglog(q_mult*qmod_mills[q_mills_start_2:q_mills_end_2].values,
               qobs_mills.flow_cms[q_mills_start_2:q_mills_end_2].values,
               'bo',markersize=1)
    plt.loglog([8*10**-1, 500], [8*10**-1, 500],'k-')
    plt.xlabel('Modeled (cms)',fontsize=12)
    plt.ylabel('Observed (cms)',fontsize=12)
    plt.title('Observed vs. Modeled Daily Streamflow\nat Lake Mills gage',fontsize=14)
    plt.ylim([8*10**-1, 500])
    plt.xlim([8*10**-1, 500])
    
    # Compute statistics
    obs=np.concatenate([qobs_mills.flow_cms[q_mills_start_1:q_mills_end_1].values,
                   qobs_mills.flow_cms[q_mills_start_2:q_mills_end_2].values])
    
    mod=np.concatenate([q_mult*qmod_mills[q_mills_start_1:q_mills_end_1].values.astype(float),
                   q_mult*qmod_mills[q_mills_start_2:q_mills_end_2].values.astype(float)])
    
    q_NSE, q_r2=compute_NSE_rs (mod, obs)


# Setup network file
if process_network=='yes':
    network=setup_network(model_input_folder)
    
    # Fix lengths that I messed up
    network.loc[122,'segment_length_m']=1680.80736
    network.loc[130,'segment_length_m']=4345.21861
    network.loc[136,'segment_length_m']=3859.81277
    network.loc[144,'segment_length_m']=939.41125
    network.loc[145,'segment_length_m']=2537.93939
    network.loc[151,'segment_length_m']=3737.05627
    
    # Fix LM gage slope- computed from 2014 DEM
    network.loc[144,'segment_slope']=0.010855
    
    # Downstream links and distances
    ds=pd.DataFrame(index=network.index, columns=['ds_strms','ds_dist_array','ds_dist_m']) # establish array of downstream stream numbers
    strm_orders_rev=np.unique(network.segment_order)[::-1] # get all segement orders in the watershed and sort from highest to lowest
    for st_o in strm_orders_rev:
        for i in network.index:
            if network.segment_order.loc[i]==st_o: # go in order of stream orders
                j=network.loc[i, 'dest_channel_id']
                if j in network.index:
                    ds.loc[i,'ds_strms']=np.append(j, ds.loc[j,'ds_strms'])
                    ds.loc[i,'ds_dist_array']=np.append(network.loc[j, 'segment_length_m'], ds.loc[j,'ds_strms'])
                    ds.loc[i,'ds_dist_m']=np.nansum(ds.loc[i,'ds_dist_array'])
                else:
                    ds.loc[i,'ds_strms']=[]
                    ds.loc[i,'ds_dist_array']=[]
                    ds.loc[i,'ds_dist_m']=0
    
    network=pd.concat([network, ds], axis=1, join_axes=[network.index]) # add these to network array 
    network[['ds_dist_m']]=network[['ds_dist_m']].astype(float)
    network = network.loc[:,~network.columns.duplicated()] # Ensure that there are no duplicate columns!
    
    # Stream Width- assume constant
    width=pd.DataFrame(index=network.index, columns=['width'],
                       data=w_const_ref*((network['total_ca_mean'].values)**exp_w_us)/(((network['total_ca_mean'][ref_stream])**exp_w_us)))
    network=pd.concat([network, width],axis=1, join_axes=[network.index]) # add these to network array
    
    # Grain Size
    os.chdir(model_data_folder)
    strm_link_vals=pickle.load(open('strm_link_vals.py', 'rb')) # stream link values of interest
    n_strms=len(strm_link_vals) # number of streams/tributary areas
    strm_orders=np.unique(network.segment_order)
    
    # Compute grain size constant at LM gage
    ### 0.005 is the slope of the LM gage reach when grain size measurements collected
    c_grain_d90=((network.total_ca_mean[ref_stream]**0.4)*0.005)/d90ch_LM     
    c_grain_dgravel=((network.total_ca_mean[ref_stream]**0.4)*0.005)/dgravelch_LM     
    c_grain_dmean=((network.total_ca_mean[ref_stream]**0.4)*0.005)/dmeanch_LM    
      
    # Compute upstream grain sizes using constant
    d90_ch_us=((network.total_ca_mean**0.4)*network.segment_slope)/c_grain_d90 
    dgravel_ch_us=((network.total_ca_mean**0.4)*network.segment_slope)/c_grain_dgravel
    dmean_ch_us=((network.total_ca_mean**0.4)*network.segment_slope)/c_grain_dmean
    # Assume sand grain size remains constant
    dsand_ch_us=dsandch_LM+network.segment_slope*0
    # Assume initial Fs is same as at LM gage- okay size will have spin up
    Ch_Fs_streams=Ch_Fs_LM+network.segment_slope*0 
    Ch_Fg_streams=(1-Ch_Fs_LM)+network.segment_slope*0 
    
    # Roughness
    if 'Qmod_median' not in locals():
        os.chdir(model_data_folder)
        Qmod_median=pickle.load(open('Qmod_median.py', 'rb'))
    H_median_lm=a_d_ref*(Qmod_median[ref_stream])**b_d_ref   
    H_median_all=H_median_lm*((network['total_ca_mean'].values)**exp_d_us)/(((network['total_ca_mean'][ref_stream])**exp_d_us))
    ng_bar=(1/(np.sqrt(8*g)))*(H_median_all**(1/6))/(1.26-2.16*np.log10(d90_ch_us/H_median_all))
    
    
    network=pd.concat([network,pd.DataFrame({"dsand_ch_m": dsand_ch_us,
                                          "dgravel_ch_m": dgravel_ch_us,
                                          "dmean_ch_m": dmean_ch_us,
                                          "Ch_Fs_strms":Ch_Fs_streams, 
                                          "ng_bar":ng_bar},index=strm_link_vals)], axis=1, join_axes=[network.index]) 
    
    network=network.drop(columns=['ds_strms','ds_dist_array','ds_dist_m'])
    network=network.loc[strm_link_vals,:]
    del(dsand_ch_us, dgravel_ch_us, dmean_ch_us)
    os.chdir(model_data_folder)
    pickle.dump(network, open("network.py", "wb"))

#%% Upload model data  
os.chdir(model_data_folder)
network=pickle.load(open('network.py', 'rb'))
streamflow=q_mult*pickle.load(open('streamflow.py', 'rb'))
strm_link_vals=pickle.load(open('strm_link_vals.py', 'rb')) # stream link values of interest

# Adjust stream segment lengths to account for the stream length bypass threshold
short_streams_id=network.index[network.segment_length_m<bypass_threshold]
short_streams_ds_id=network.loc[short_streams_id,'dest_channel_id']
short_streams_len=network.loc[short_streams_id,'segment_length_m'].values
short_streams_ds_len=network.loc[short_streams_ds_id,'segment_length_m'].values
network.loc[short_streams_ds_id,'segment_length_m']=short_streams_len+short_streams_ds_len
#%%
# INITIAL COMPUTATIONS/VARIABLE ALLOCATIONS
n_strms=len(strm_link_vals) # number of streams/tributary areas
strm_orders=np.unique(network.segment_order) # array of unique stream orders
if start_date != '1/1/1915':
    streamflow=streamflow.loc[start_date::,:]
start_year_len=len(streamflow.index[streamflow.index.year==streamflow.index[0].year]) # number of days in first year

# Preallocate sediment volume data frames representing sediment "buckets" at the end of each timestep (day)
# Vg=gravel; Vs=sand; Vm=mud (silt, clay)
# Volumes stored on bed [m3]
Vg_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Cumulative Volumes stored on bed [m3]
Vg_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volumes deposited from mass wasting event [m3]
Vg_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volume of bedload that stream had capacity to transport  [m3]
Vg_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vsb_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vss_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volume transported out of stream  [m3]
Vg_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vsb_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vss_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Lake Mills gage rating curve
LM_gage_rc=pd.DataFrame(0, index=np.arange (0,len(streamflow.index)), columns=['g','sb','ss','m'])


# Other
Ch_Fs=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Ch_Fg=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#C_ss=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#Vs_s=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#Vs_e=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#Vs_s_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
#Vm_e_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))

if recycle_run=='yes':
    os.chdir(recycle_run_directory)
    #Set any initial (time=0) conditions in streams
    # Volumes stored on bed [m3]
    Vg_b.loc[:,0]=np.load('Vg_b.npy')[:,-1]
    Vs_b.loc[:,0]=np.load('Vs_b.npy')[:,-1]
    Vm_b.loc[:,0]=np.load('Vm_b.npy')[:,-1]
    # Volumes deposited from mass wasting event [m3]
    Vg_mw.loc[:,0]=np.load('Vg_mw.npy')[:,-1]
    Vs_mw.loc[:,0]=np.load('Vs_mw.npy')[:,-1]
    Vm_mw.loc[:,0]=np.load('Vm_mw.npy')[:,-1]
    # Volume that stream had capacity to transport  [m3]
    Vg_cap.loc[:,0]=0
    Vsb_cap.loc[:,0]=0
    Vss_cap.loc[:,0]=0
    # Volume transported out of stream  [m3]
    Vg_t.loc[:,0]=np.load('Vg_t.npy')[:,-1]
    Vsb_t.loc[:,0]=np.load('Vsb_t.npy')[:,-1]
    Vss_t.loc[:,0]=np.load('Vss_t.npy')[:,-1]
    Vm_t.loc[:,0]=np.load('Vm_t.npy')[:,-1]
    
    # Set arrays for times between mass wasting events
    time_mw=pd.DataFrame(data=np.load('time_mw.npy'), index=strm_link_vals, columns=['output']) # time since last mass wasting event for each stream link
    tL_mw=pd.DataFrame(data=np.load('tL_mw.npy'), index=strm_link_vals, columns=['output']) # turn into a data frame that will grow
    
    Ch_Fs.loc[:,0]=np.load('Ch_Fs.npy')[:,-1]
    Ch_Fg.loc[:,0]=np.load('Ch_Fg.npy')[:,-1]
#    C_ss.loc[:,0]=np.load('C_ss.npy')[:,-1]
#    Vs_s.loc[:,0]=np.load('Vs_s.npy')[:,-1]
#    Vs_e.loc[:,0]=np.load('Vs_e.npy')[:,-1]

    np.unique(streamflow.index.year)[0]
    
    # Annually- Preallocate sediment volume data frames representing sediment "buckets" for year
    # Volumes stored on bed [m3]
    Vg_b_annual=pd.DataFrame(data=np.load('Vg_b_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vs_b_annual=pd.DataFrame(data=np.load('Vs_b_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vm_b_annual=pd.DataFrame(data=np.load('Vm_b_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    
    # Cumulative Volumes stored on bed [m3]
    Vg_b_ctv_annual=pd.DataFrame(data=np.load('Vg_b_ctv_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vs_b_ctv_annual=pd.DataFrame(data=np.load('Vs_b_ctv_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vm_b_ctv_annual=pd.DataFrame(data=np.load('Vm_b_ctv_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    
    # Volumes deposited from mass wasting event [m3]
    Vg_mw_annual=pd.DataFrame(data=np.load('Vg_mw_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vs_mw_annual=pd.DataFrame(data=np.load('Vs_mw_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vm_mw_annual=pd.DataFrame(data=np.load('Vm_mw_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    # Volume of bedload  that stream had capacity to transport  [m3]
    Vg_cap_annual=pd.DataFrame(data=np.load('Vg_cap_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vsb_cap_annual=pd.DataFrame(data=np.load('Vsb_cap_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vss_cap_annual=pd.DataFrame(data=np.load('Vss_cap_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    # Volume transported out of stream  [m3]
    Vg_t_annual=pd.DataFrame(data=np.load('Vg_t_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vsb_t_annual=pd.DataFrame(data=np.load('Vsb_t_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vss_t_annual=pd.DataFrame(data=np.load('Vss_t_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))
    Vm_t_annual=pd.DataFrame(data=np.load('Vm_t_annual.npy'),index=strm_link_vals, columns=np.arange(1915,2012))

if recycle_run=='no':
    #Set any initial (time=0) conditions in streams
    # Volumes stored on bed [m3]
    Vg_b.loc[:,0]=0.8*(1-network['Ch_Fs_strms'])*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
    Vs_b.loc[:,0]=0.8*network['Ch_Fs_strms']*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
    Vm_b.loc[:,0]=0.2*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
    # Volumes deposited from mass wasting event [m3]
    Vg_mw.loc[:,0]=0
    Vs_mw.loc[:,0]=0
    Vm_mw.loc[:,0]=0
    # Volume that stream had capacity to transport  [m3]
    Vg_cap.loc[:,0]=0
    Vsb_cap.loc[:,0]=0
    Vss_cap.loc[:,0]=0
    # Volume transported out of stream  [m3]
    Vg_t.loc[:,0]=0
    Vsb_t.loc[:,0]=0
    Vss_t.loc[:,0]=0
    Vm_t.loc[:,0]=0
    
    # Set arrays for times between mass wasting events
    time_mw=pd.DataFrame(0, index=strm_link_vals, columns=['output']) # time since last mass wasting event for each stream link
    tL_mw=np.ceil(np.random.exponential(scale=beta_mw, size=len(strm_link_vals))) # time (years) until first mass wasting event- randomly sampled from exponential distribution
    tL_mw=pd.DataFrame(data=tL_mw, index=strm_link_vals, columns=['output']) # turn into a data frame that will grow
    
    # Other- temporary
    Ch_Fs=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
    Ch_Fs.loc[:,0]=network['Ch_Fs_strms']
    Ch_Fg=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
    Ch_Fg.loc[:,0]=1-network['Ch_Fs_strms']
#    C_ss=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#    Vs_s=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#    Vs_e=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
#    
#    Vs_s_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
#    Vm_e_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    
#    C_ss.loc[:,0]=0
#    Vs_s.loc[:,0]=0
#    Vs_e.loc[:,0]=0

    # Annually- Preallocate sediment volume data frames representing sediment "buckets" for year
    # Volumes stored on bed [m3]
    Vg_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vs_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vm_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    
    # Cumulative Volumes stored on bed [m3]
    Vg_b_ctv_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vs_b_ctv_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vm_b_ctv_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    
    # Volumes deposited from mass wasting event [m3]
    Vg_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vs_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vm_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    # Volume of bedload  that stream had capacity to transport  [m3]
    Vg_cap_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year)) 
    Vsb_cap_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vss_cap_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    # Volume transported out of stream  [m3]
    Vg_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vsb_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vss_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
    Vm_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))

#%%
start = time.time()
date_string=time.strftime("%Y-%m-%d_%H%M")
os.chdir(model_output_folder)
os.mkdir(date_string)
os.chdir(model_output_folder+date_string)
day_of_year=0
## RUN MODEL
for i in range (0,len(streamflow.index)):
    print(streamflow.index[i])
    date=streamflow.index[i]
    day_of_year=day_of_year+1
    time_mw=time_mw+1 # start new year
   # run stochastic mass wastinig generator for all stream links
    for st_o in strm_orders:
        for j in network.index.values:
            if network.segment_order.loc[j]==st_o: # go in order of stream orders
                if time_mw.loc[j,'output']==tL_mw.loc[j,'output']: # if time since the last mass wasting event is the same as randomly generated lag time
                    # Generate mass wasting volumes based on denudation rate
                    if mw_composition=='stochastic':
                        # Generate random values for percent gravel, sand, and mud
                        temp_rand=np.random.rand(3)
                        mw_pcnts=temp_rand/sum(temp_rand)
                        # Generate mass wasting volumes based on denudation rate
                        Vg_mw.loc[j,day_of_year]=mw_pcnts[0]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']
                        Vs_mw.loc[j,day_of_year]=mw_pcnts[1]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                   
                        Vm_mw.loc[j,day_of_year]=mw_pcnts[2]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                      
                    if mw_composition=='set':
                        Vg_mw.loc[j,day_of_year]=pct_g_mw_dep*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']
                        Vs_mw.loc[j,day_of_year]=pct_s_mw_dep*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                   
                        Vm_mw.loc[j,day_of_year]=pct_m_mw_dep*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                   
                    # Update mass wasting clocks
                    time_mw.loc[j,'output']=0 # Reset counter until next mass wasting event
                    tL_mw.loc[j,'output']=np.ceil(np.random.exponential(scale=beta_mw))  # generate a new lag time until next mass wasting event and add it to array of lag times#
                else: # If have not reach the lag time for a mass wasting event at the stream then the volume will be the same as last year
                    # Not mass wasting volume is added
                    Vg_mw.loc[j,day_of_year]=0
                    Vs_mw.loc[j,day_of_year]=0
                    Vm_mw.loc[j,day_of_year]=0

                if network.segment_length_m.loc[j]>=bypass_threshold:
                # Add mass wasting volume to the bed from previous timestep
                    Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year-1]+Vg_mw.loc[j,day_of_year]
                    
                    Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year-1]+(1-pct_ss_mw)*Vs_mw.loc[j,day_of_year]
                    Vss_t.loc[j,day_of_year]=pct_ss_mw*Vs_mw.loc[j,day_of_year]
                    
                    Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year-1]+(1-pct_m_mw)*Vm_mw.loc[j,day_of_year] 
                    Vm_t.loc[j,day_of_year]=pct_m_mw*Vm_mw.loc[j,day_of_year] 
    
                    # Add in sediment from upstream tributaries
                    if st_o!=1: # if not at a headwater stream (i.e., stream order >1), need to add volume of sediment coming from upstream
                        feeder_links=network.loc[network['dest_channel_id']==j].index.values # find the stream links that feed into the link which are immediately upstream
                        # sum up the volume transported out of the upstream link and add to current volume
                        # bedload
                        Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year]+np.nansum(Vg_t.loc[feeder_links,day_of_year])  
                        Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]+np.nansum(Vsb_t.loc[feeder_links,day_of_year])
                        # suspended load
                        Vss_t.loc[j,day_of_year]=Vss_t.loc[j,day_of_year]+np.nansum(Vss_t.loc[feeder_links,day_of_year])
                        Vm_t.loc[j,day_of_year]=Vm_t.loc[j,day_of_year]+np.nansum(Vm_t.loc[feeder_links,day_of_year])
                    
                    # Compute shear stress and shear velocity from modeled flow
                    Q=streamflow.loc[date,j]
                    Qvol=Q*3600*24 # m3 of streamflow throughout the day [m3]
                    tau_strm, u_strm= compute_channel_properties(ref_stream, a_u_ref, b_u_ref, a_d_ref, b_d_ref, a_n, b_n, network['ng_bar'][j], 
                                                                 network.segment_slope[j], network['total_ca_mean'][ref_stream], network['total_ca_mean'][j], 
                                                                 network['width'][j], streamflow.loc[date,ref_stream], Q)
    
                    # Suspended Sediment- Based on Patil et al., 2012
                    SS_vol=(Vss_t.loc[j,day_of_year]+Vm_t.loc[j,day_of_year])
                    C_ss=sg*rho_w*SS_vol/(SS_vol+Qvol)
                    if Vg_b.loc[j,day_of_year]==0: # if there's no surface sediment then no transport
                        Ch_Fg_temp=0           
                    else: 
                        Ch_Fg_temp=(Vg_b.loc[j,day_of_year])/(Vg_b.loc[j,day_of_year]+Vs_b.loc[j,day_of_year]+Vm_b.loc[j,day_of_year])
                   
                    if C_ss>0.65*sg*rho_w: # the maximum volumetric concentration is 0.65
                        C_ss=0.65*sg*rho_w
                    # Suspended sediment- silt: Compute deposition and transport
                    # Vm_d is deposition from C_ss to the bed
                    if tau_strm>tau_c_fines: # is tau exceeds critical tau for fines, erode/transport whatever mud is available
                        Vm_e=min((1-Ch_Fg_temp)*pcnt_m_er*Vm_b.loc[j,day_of_year], (1-Ch_Fg_temp)*pcnt_m_er*(0.0002)*((tau_strm/tau_c_fines)-1)*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24/(sg*rho_w)) # no mud is deposited, all mud is transported
                        Vm_t.loc[j,day_of_year]=Vm_t.loc[j,day_of_year]+Vm_e
                        Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year]-Vm_e
                    else: # zone concentration limits (c1_m and c2_m)
                        Vm_e=0
                        if C_ss<=c1_m:
                            v_ss_m=a_w_m*(c1_m**n_w_m)/((c1_m**2+b_w_m**2)**m_w_m)
                        elif C_ss>c2_m:
                            v_ss_m=a_w_m*(c2_m**n_w_m)/((c2_m**2+b_w_m**2)**m_w_m)
                        else:
                            v_ss_m=a_w_m*(C_ss**n_w_m)/((C_ss**2+b_w_m**2)**m_w_m)                        
                        Vm_d=(1-(tau_strm/tau_c_fines))*v_ss_m*C_ss*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24/(sg*rho_w)
                        if Vm_d<0:
                            print('Error: Vm_d<0 for i,j=',np.array([i,j]))
                        if Vm_d<Vm_t.loc[j,day_of_year]:
                            Vm_t.loc[j,day_of_year]=Vm_t.loc[j,day_of_year]-Vm_d
                            Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year]+Vm_d
                        else: # Vm_d>Vm_t.loc[j,day_of_year]
                            Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year]+Vm_t.loc[j,day_of_year]
                            Vm_t.loc[j,day_of_year]=0
                    
                    # Suspended sediment- sand: Compute deposition/erosion
                    # Compute deposition
                    Z_R=v_ss/(0.41*u_strm) # Rouse number
                    int_Z_R=1/(c0_ss+c1_ss*Z_R+c2_ss*Z_R**2+c3_ss*Z_R**3+c4_ss*Z_R**4+c5_ss*Z_R**5+c6_ss*Z_R**6)
                    Vs_s=v_ss*C_ss*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24/(int_Z_R*sg*rho_w) #m3/day- volume of deposited sand
                    # Compute erosion
                    Zu=alp_1*u_strm*(Re_p**alp_2)/v_sb
                    Vs_e=(1-Ch_Fg_temp)*pcnt_sb_er*(v_sb*A_ss*Zu**5)*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24/(1+(A_ss*Zu**5)/0.3) #m3/day
                    # Compute net suspended sediment
                    Vss_cap.loc[j,day_of_year]=max(0,Vs_e-Vs_s)
                    
                    if Vss_t.loc[j,day_of_year]>Vss_cap.loc[j,day_of_year]:
                        b_ss_transfer=0
                        Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]+(Vss_t.loc[j,day_of_year]-Vss_cap.loc[j,day_of_year])
                        Vss_t.loc[j,day_of_year]=Vss_cap.loc[j,day_of_year]
                    
                    if Vss_t.loc[j,day_of_year]<Vss_cap.loc[j,day_of_year]:
                        b_ss_transfer=min((1-Ch_Fg_temp)*pcnt_sb_er*Vs_b.loc[j,day_of_year], Vss_cap.loc[j,day_of_year]-Vss_t.loc[j,day_of_year])
                        Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]-b_ss_transfer
                        Vss_t.loc[j,day_of_year]=Vss_t.loc[j,day_of_year]+b_ss_transfer
                                 
                    # Recompute W&C 2003 parameterizaion of sand and gravel on the channel bed
                    if Vs_b.loc[j,day_of_year]==0 and Vg_b.loc[j,day_of_year]==0: # if there's no surface sediment then no transport
                        Ch_Fs.loc[j,day_of_year]=0
                        Ch_Fg.loc[j,day_of_year]=0
                        Vg_cap.loc[j,day_of_year]=0 
                        Vg_t.loc[j,day_of_year]=0 
                        Vg_b.loc[j,day_of_year]=0 
                        Vsb_cap.loc[j,day_of_year]=0 
                        Vsb_t.loc[j,day_of_year]=0
                        Vs_b.loc[j,day_of_year]=0 
                        
                    else: # if all good!
                        Ch_Fs.loc[j,day_of_year]=(Vs_b.loc[j,day_of_year])/(Vs_b.loc[j,day_of_year]+Vg_b.loc[j,day_of_year])
                        Ch_Fg.loc[j,day_of_year]=(Vg_b.loc[j,day_of_year])/(Vs_b.loc[j,day_of_year]+Vg_b.loc[j,day_of_year])
#                        Ch_Fs.loc[j,day_of_year]=(Vs_b.loc[j,day_of_year])/(Vm_e+b_ss_transfer+Vs_b.loc[j,day_of_year]+Vg_b.loc[j,day_of_year])
#                        Ch_Fg.loc[j,day_of_year]=(Vg_b.loc[j,day_of_year])/(Vm_e+b_ss_transfer+Vs_b.loc[j,day_of_year]+Vg_b.loc[j,day_of_year])
                        
                        tau_star_rsm=0.021+0.015*np.exp(-20*Ch_Fs.loc[j,day_of_year]) # dimensionless reference shear stress for mean grain size
                        tau_rsm=tau_star_rsm*(sg-1)*rho_w*g*(network.loc[j,'dmean_ch_m']) # reference shear stress for mean grain size [N/m2]
                        b_sand=0.67/(1+np.exp(1.5-(network.loc[j,'dsand_ch_m']/network.loc[j,'dmean_ch_m']))) # b parameter for sand
                        b_gravel=0.67/(1+np.exp(1.5-(network.loc[j,'dgravel_ch_m']/network.loc[j,'dmean_ch_m']))) # b parameter for gravel
                        tau_r_sand=tau_rsm*(network.loc[j,'dsand_ch_m']/network.loc[j,'dmean_ch_m'])**b_sand # reference tau for sand [N/m2]
                        tau_r_gravel=tau_rsm*(network.loc[j,'dgravel_ch_m']/network.loc[j,'dmean_ch_m'])**b_gravel # reference tau for gravel [N/m2]
                                 
                        # Compute Bedload sediment transport capacity with calibrated Wilcock and Crowe equation
                        Wstar_gravel, Wstar_sand=run_wc2003_2F_model (tau_strm, tau_r_sand, tau_r_gravel)
                        
                        # Compute gravel transport capacity, volume transported, and volume remaining
                        Vg_cap.loc[j,day_of_year]=3600*24*((u_strm)**3)*Wstar_gravel*Ch_Fg.loc[j,day_of_year]/((sg-1)*g)*network.loc[j,'width'] #  m3 (unit is m3/day; timestep is 1 day)
                        Vg_t.loc[j,day_of_year]=np.min([Vg_cap.loc[j,day_of_year],Vg_b.loc[j,day_of_year]]) # gravel: volume transported is the minimum between available volume and capacity of flow
                        Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year]-Vg_t.loc[j,day_of_year] # new volume deposited in stream- difference between existing volume and volume transported
        
                        # Compute sand transport capacity, volume transported, and volume remaining
                        Vsb_cap.loc[j,day_of_year]=3600*24*((u_strm)**3)*Wstar_sand*Ch_Fs.loc[j,day_of_year]/((sg-1)*g)*network.loc[j,'width'] #  m3 (unit is m3/day; timestep is 1 day)
                        Vsb_t.loc[j,day_of_year]=np.min([Vsb_cap.loc[j,day_of_year],Vs_b.loc[j,day_of_year]]) # gravel: volume transported is the minimum between available volume and capacity of flow
                        Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]-Vsb_t.loc[j,day_of_year] # new volume deposited in stream- difference between existing volume and volume transported
        
                    # Cumulative volume
                    if Vg_b.loc[j,day_of_year]>Vg_b.loc[j,day_of_year-1]:
                        Vg_b_ctv.loc[j,day_of_year]=Vg_b_ctv.loc[j,day_of_year-1]+Vg_b.loc[j,day_of_year]-Vg_b.loc[j,day_of_year-1]
                    else:
                        Vg_b_ctv.loc[j,day_of_year]=Vg_b_ctv.loc[j,day_of_year-1]
    
                    if Vs_b.loc[j,day_of_year]>Vs_b.loc[j,day_of_year-1]:
                        Vs_b_ctv.loc[j,day_of_year]=Vs_b_ctv.loc[j,day_of_year-1]+Vs_b.loc[j,day_of_year]-Vs_b.loc[j,day_of_year-1]
                    else:
                        Vs_b_ctv.loc[j,day_of_year]=Vs_b_ctv.loc[j,day_of_year-1]
    
                    if Vm_b.loc[j,day_of_year]>Vm_b.loc[j,day_of_year-1]:
                        Vm_b_ctv.loc[j,day_of_year]=Vm_b_ctv.loc[j,day_of_year-1]+Vm_b.loc[j,day_of_year]-Vm_b.loc[j,day_of_year-1]
                    else:
                        Vm_b_ctv.loc[j,day_of_year]=Vm_b_ctv.loc[j,day_of_year-1]
                else:
                    if st_o!=1: # if not at a headwater stream (i.e., stream order >1), need to add volume of sediment coming from upstream
                        feeder_links=network.loc[network['dest_channel_id']==j].index.values # find the stream links that feed into the link which are immediately upstream
                        Vg_t.loc[j,day_of_year]=Vg_mw.loc[j,day_of_year]+np.nansum(Vg_t.loc[feeder_links,day_of_year])  
                        Vsb_t.loc[j,day_of_year]=(1-pct_ss_mw)*Vs_mw.loc[j,day_of_year]+np.nansum(Vsb_t.loc[feeder_links,day_of_year])
                        Vss_t.loc[j,day_of_year]=pct_ss_mw*Vs_mw.loc[j,day_of_year]+np.nansum(Vss_t.loc[feeder_links,day_of_year])
                        Vm_t.loc[j,day_of_year]=Vm_mw.loc[j,day_of_year] +np.nansum(Vm_t.loc[feeder_links,day_of_year])
                    else:
                        Vg_t.loc[j,day_of_year]=Vg_mw.loc[j,day_of_year]
                        Vsb_t.loc[j,day_of_year]=(1-pct_ss_mw)*Vs_mw.loc[j,day_of_year]
                        Vss_t.loc[j,day_of_year]=pct_ss_mw*Vs_mw.loc[j,day_of_year]
                        Vm_t.loc[j,day_of_year]=Vm_mw.loc[j,day_of_year]
                    Vg_b.loc[j,day_of_year]=np.nan
                    Vs_b.loc[j,day_of_year]=np.nan
                    Vm_b.loc[j,day_of_year]=np.nan            
                    Vg_b_ctv.loc[j,day_of_year]=np.nan
                    Vs_b_ctv.loc[j,day_of_year]=np.nan
                    Vm_b_ctv.loc[j,day_of_year]=np.nan
                    Vg_cap.loc[j,day_of_year]=np.nan
                    Vsb_cap.loc[j,day_of_year]=np.nan
                    Vss_cap.loc[j,day_of_year]=np.nan
                    Ch_Fs.loc[j,day_of_year]=np.nan
                    Ch_Fg.loc[j,day_of_year]=np.nan
#                    C_ss.loc[j,day_of_year]=np.nan
#                    Vs_s.loc[j,day_of_year]=np.nan
#                    Vs_e.loc[j,day_of_year]=np.nan
                    
                    
    LM_gage_rc.loc[i,'g']= Vg_t.loc[144, day_of_year]
    LM_gage_rc.loc[i,'sb']=Vsb_t.loc[144, day_of_year]
    LM_gage_rc.loc[i,'ss']=Vss_t.loc[144, day_of_year]
    LM_gage_rc.loc[i,'m']=Vm_t.loc[144, day_of_year]
    
    if date.month==12 and date.day==31 and date.year<forcing_end_date.year:
        if date.year==streamflow.index[streamflow.index.year==streamflow.index[-1].year][0].year:
            next_year_len=0
        else: next_year_len=len(streamflow.index[streamflow.index.year==date.year+1])
        
        # Daily loads
        Vg_b_last=Vg_b.loc[:, Vg_b.columns[-1]]
        Vs_b_last=Vs_b.loc[:, Vs_b.columns[-1]]
        Vm_b_last=Vm_b.loc[:, Vm_b.columns[-1]]
        
        Vg_b_ctv_last=Vg_b_ctv.loc[:, Vg_b_ctv.columns[-1]]
        Vs_b_ctv_last=Vs_b_ctv.loc[:, Vs_b_ctv.columns[-1]]
        Vm_b_ctv_last=Vm_b_ctv.loc[:, Vm_b_ctv.columns[-1]]

        Vg_mw_last=Vg_mw.loc[:, Vg_mw.columns[-1]]
        Vs_mw_last=Vs_mw.loc[:, Vs_mw.columns[-1]]
        Vm_mw_last=Vm_mw.loc[:, Vm_mw.columns[-1]]
        
        Vg_cap_last=Vg_cap.loc[:, Vg_cap.columns[-1]]
        Vsb_cap_last=Vsb_cap.loc[:, Vsb_cap.columns[-1]]
        Vss_cap_last=Vss_cap.loc[:, Vss_cap.columns[-1]]
        
        Vg_t_last=Vg_t.loc[:, Vg_t.columns[-1]]
        Vss_t_last=Vss_t.loc[:, Vss_t.columns[-1]]
        Vsb_t_last=Vsb_t.loc[:, Vsb_t.columns[-1]]
        Vm_t_last=Vm_t.loc[:, Vm_t.columns[-1]]

        Ch_Fs_last=Ch_Fs.loc[:, Ch_Fs.columns[-1]]
        Ch_Fg_last=Ch_Fg.loc[:, Ch_Fg.columns[-1]]
#        C_ss_last=C_ss.loc[:, C_ss.columns[-1]]
#        Vs_s_last=Vs_s.loc[:, Vs_s.columns[-1]]
#        Vs_e_last=Vs_s.loc[:, Vs_e.columns[-1]]

#         Monthly Loads
#        Vg_b_monthly.loc[:, date.year]=Vg_b.resample("M", axis=0).sum()
#        Vs_b_monthly.loc[:, date.year]=Vs_b_last
#        Vm_b_monthly.loc[:, date.year]=Vm_b_last
#
#        Vg_b_ctv_monthly.loc[:, date.year]=Vg_b_ctv_last
#        Vs_b_ctv_monthly.loc[:, date.year]=Vs_b_ctv_last
#        Vm_b_ctv_monthly.loc[:, date.year]=Vm_b_ctv_last
#
#        Vg_mw_monthly.loc[:, date.year]=Vg_mw[1::].sum(axis=1)
#        Vs_mw_monthly.loc[:, date.year]=Vs_mw[1::].sum(axis=1)
#        Vm_mw_monthly.loc[:, date.year]=Vm_mw[1::].sum(axis=1)
#
#        Vg_cap_monthly.loc[:, date.year]=Vg_cap[1::].sum(axis=1)
#        Vsb_cap_monthly.loc[:, date.year]=Vsb_cap[1::].sum(axis=1)
#        Vss_cap_monthly.loc[:, date.year]=Vss_cap[1::].sum(axis=1)
#
#        Vg_t_monthly.loc[:, date.year]=Vg_t[1::].sum(axis=1)
#        Vsb_t_monthly.loc[:, date.year]=Vsb_t[1::].sum(axis=1)
#        Vss_t_monthly.loc[:, date.year]=Vss_t[1::].sum(axis=1)
#        Vm_t_monthly.loc[:, date.year]=Vm_t[1::].sum(axis=1)

        # Annual loads
        Vg_b_annual.loc[:, date.year]=Vg_b_last
        Vs_b_annual.loc[:, date.year]=Vs_b_last
        Vm_b_annual.loc[:, date.year]=Vm_b_last

        Vg_b_ctv_annual.loc[:, date.year]=Vg_b_ctv_last
        Vs_b_ctv_annual.loc[:, date.year]=Vs_b_ctv_last
        Vm_b_ctv_annual.loc[:, date.year]=Vm_b_ctv_last

        Vg_mw_annual.loc[:, date.year]=Vg_mw[1::].sum(axis=1)
        Vs_mw_annual.loc[:, date.year]=Vs_mw[1::].sum(axis=1)
        Vm_mw_annual.loc[:, date.year]=Vm_mw[1::].sum(axis=1)

        Vg_cap_annual.loc[:, date.year]=Vg_cap[1::].sum(axis=1)
        Vsb_cap_annual.loc[:, date.year]=Vsb_cap[1::].sum(axis=1)
        Vss_cap_annual.loc[:, date.year]=Vss_cap[1::].sum(axis=1)

        Vg_t_annual.loc[:, date.year]=Vg_t[1::].sum(axis=1)
        Vsb_t_annual.loc[:, date.year]=Vsb_t[1::].sum(axis=1)
        Vss_t_annual.loc[:, date.year]=Vss_t[1::].sum(axis=1)
        Vm_t_annual.loc[:, date.year]=Vm_t[1::].sum(axis=1)
        
        np.save('Vg_b_ctv_annual', Vg_b_ctv_annual)
        np.save('Vs_b_ctv_annual', Vs_b_ctv_annual)
        np.save('Vm_b_ctv_annual', Vm_b_ctv_annual)

        np.save('Vg_b_annual', Vg_b_annual)
        np.save('Vs_b_annual', Vs_b_annual)
        np.save('Vm_b_annual', Vm_b_annual)
        
        np.save('Vg_mw_annual', Vg_mw_annual)   
        np.save('Vs_mw_annual', Vs_mw_annual)
        np.save('Vm_mw_annual', Vm_mw_annual)   

        np.save('Vg_cap_annual', Vg_cap_annual)
        np.save('Vsb_cap_annual', Vsb_cap_annual)
        np.save('Vss_cap_annual', Vss_cap_annual)

        np.save('Vg_t_annual', Vg_t_annual)
        np.save('Vsb_t_annual', Vsb_t_annual)
        np.save('Vss_t_annual', Vss_t_annual)
        np.save('Vm_t_annual', Vm_t_annual)
        
        np.save('Vg_b', Vg_b)
        np.save('Vs_b', Vs_b)
        np.save('Vm_b', Vm_b)
        
        np.save('Vg_b_ctv', Vg_b_ctv)
        np.save('Vs_b_ctv', Vs_b_ctv)
        np.save('Vm_b_ctv', Vm_b_ctv)

        np.save('Vg_mw', Vg_mw)   
        np.save('Vs_mw', Vs_mw)
        np.save('Vm_mw', Vm_mw)   

        np.save('Vg_cap', Vg_cap)
        np.save('Vsb_cap', Vsb_cap)
        np.save('Vss_cap', Vss_cap)

        np.save('Vg_t', Vg_t)
        np.save('Vsb_t', Vsb_t)
        np.save('Vss_t', Vss_t)
        np.save('Vm_t', Vm_t)

        np.save('Ch_Fs',  Ch_Fs)
        np.save('Ch_Fg',  Ch_Fg)
#        np.save('C_ss',  C_ss)
#        np.save('Vs_s',  Vs_s)
#        np.save('Vs_e',  Vs_e)
        np.save('time_mw', time_mw)
        np.save('tL_mw', tL_mw)
        np.save('LM_gage_rc',LM_gage_rc)

        Vg_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vm_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
    
        Vg_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vm_b_ctv=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))

        Vg_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vm_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))

        Vg_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vsb_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vss_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))

        Vg_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vsb_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vss_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vm_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))

        Ch_Fs=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Ch_Fg=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
#        C_ss=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
#        Vs_s=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
#        Vs_e=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))

        Vg_b.loc[:, Vg_b.columns[0]]=Vg_b_last
        Vs_b.loc[:, Vs_b.columns[0]]=Vs_b_last
        Vm_b.loc[:, Vm_b.columns[0]]=Vm_b_last

        Vg_b_ctv.loc[:, Vg_b_ctv.columns[0]]=Vg_b_ctv_last
        Vs_b_ctv.loc[:, Vs_b_ctv.columns[0]]=Vs_b_ctv_last
        Vm_b_ctv.loc[:, Vm_b_ctv.columns[0]]=Vm_b_ctv_last
        
        Vg_mw.loc[:, Vg_mw.columns[0]]=Vg_mw_last
        Vs_mw.loc[:, Vs_mw.columns[0]]=Vs_mw_last
        Vm_mw.loc[:, Vm_mw.columns[0]]=Vm_mw_last

        Vg_cap.loc[:, Vg_cap.columns[0]]=Vg_cap_last
        Vsb_cap.loc[:, Vsb_cap.columns[0]]=Vsb_cap_last
        Vss_cap.loc[:, Vss_cap.columns[0]]=Vss_cap_last

        Vg_t.loc[:, Vg_t.columns[0]]=Vg_t_last
        Vsb_t.loc[:, Vsb_t.columns[0]]=Vsb_t_last
        Vss_t.loc[:, Vss_t.columns[0]]=Vss_t_last
        Vm_t.loc[:, Vm_t.columns[0]]=Vm_t_last

        Ch_Fs.loc[:, Ch_Fs.columns[0]]=Ch_Fs_last
        Ch_Fg.loc[:, Ch_Fg.columns[0]]=Ch_Fg_last
#        C_ss.loc[:, C_ss.columns[0]]=C_ss_last
#        Vs_s.loc[:, Vs_s.columns[0]]=Vs_s_last
#        Vs_e.loc[:, Vs_e.columns[0]]=Vs_e_last        
        day_of_year=0
        
# Save results!
end = time.time()
print(end - start)

np.save('Vg_b_annual', Vg_b_annual)
np.save('Vs_b_annual', Vs_b_annual)
np.save('Vm_b_annual', Vm_b_annual)

np.save('Vg_b_ctv_annual', Vg_b_ctv_annual)
np.save('Vs_b_ctv_annual', Vs_b_ctv_annual)
np.save('Vm_b_ctv_annual', Vm_b_ctv_annual)

np.save('Vg_mw_annual', Vg_mw_annual)
np.save('Vs_mw_annual', Vs_mw_annual)
np.save('Vm_mw_annual', Vm_mw_annual)

np.save('Vg_cap_annual', Vg_cap_annual)
np.save('Vsb_cap_annual', Vsb_cap_annual)
np.save('Vss_cap_annual', Vss_cap_annual)

np.save('Vg_t_annual', Vg_t_annual)
np.save('Vsb_t_annual', Vsb_t_annual)
np.save('Vss_t_annual', Vss_t_annual)
np.save('Vm_t_annual', Vm_t_annual)

np.save('Ch_Fs',  Ch_Fs)
np.save('Ch_Fg',  Ch_Fg)
#np.save('C_ss',  C_ss)
#np.save('Vs_s',  Vs_s)
#np.save('Vs_e',  Vs_e)
np.save('time_mw', time_mw)
np.save('tL_mw', tL_mw)

np.save('Vg_b', Vg_b)
np.save('Vs_b', Vs_b)
np.save('Vm_b', Vm_b)

np.save('Vg_b_ctv', Vg_b_ctv)
np.save('Vs_b_ctv', Vs_b_ctv)
np.save('Vm_b_ctv', Vm_b_ctv)

np.save('Vg_mw', Vg_mw)
np.save('Vs_mw', Vs_mw)
np.save('Vm_mw', Vm_mw)

np.save('Vg_cap', Vg_cap)
np.save('Vsb_cap', Vsb_cap)
np.save('Vss_cap', Vss_cap)

np.save('Vg_t', Vg_t)
np.save('Vsb_t', Vsb_t)
np.save('Vss_t', Vss_t)
np.save('Vm_t', Vm_t)

#np.save('Vs_e', Vs_e)
#np.save('Vs_s', Vs_s)

#%%
streams_to_plot=[103, 144, 157, 251, 304, 387, 403] # [103, 144, 251, 284, 291, 304, 309, 310, 338, 358, 370, 377, 381, 403]
# Daily review for last year
daily_start_date=datetime.date(1915,1,1) # for daily review- start the day before year of interest
daily_end_date=datetime.date(1915,9,12)# for daily review

for stream_index in streams_to_plot:  
    plot_title=''
    fig, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5,1, sharex=True, sharey=False)        
    
    ax1.plot(Vg_mw.columns,
             Vg_mw.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-', linewidth=3, label='gravel')
    ax1.plot(Vg_mw.columns,
             Vs_mw.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3, label='sand')
    ax1.plot(Vg_mw.columns,
             Vm_mw.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g-', linewidth=3, label='mud')
    ax1.set_title('Stream Number '+str(stream_index)+'\nSediment Deposited from Landslide',fontsize=10)
    ax1.set_ylabel('Depth (m)',fontsize=8)
    ax1.legend()
    
    ax2.plot(Vg_mw.columns, 
             Vg_cap.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-', linewidth=3)  
    ax2.plot(Vg_mw.columns, 
             Vsb_cap.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3)
    ax2.set_title('Bedload Transport Capacity of Stream',fontsize=10)
    ax2.set_ylabel('Depth (m)',fontsize=8)
    #ax2.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax3.plot(Vg_mw.columns,
             Vg_t.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-',linewidth=3)
    ax3.plot(Vg_mw.columns,
             Vsb_t.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-',linewidth=3)
    ax3.plot(Vg_mw.columns,
             Vss_t.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'c-',linewidth=3)
    ax3.plot(Vg_mw.columns,
             Vm_t.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g-',linewidth=3)
    ax3.set_title('Sediment Transported Out of Channel',fontsize=10)
    ax3.set_ylabel('Depth (m)',fontsize=8)
    #ax3.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax4.plot(Vg_mw.columns,
             Vg_b.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-', linewidth=3)
    ax4.plot(Vg_mw.columns,
             Vs_b.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3)
    ax4.plot(Vg_mw.columns,
             Vm_b.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g-', linewidth=3)
    ax4.set_title('Sediment Accumulated on Channel Bed',fontsize=10)
    ax4.set_ylabel('Depth (m)',fontsize=8)
    
    ax5.plot(Vg_mw.columns,
             Vg_b_ctv.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-', linewidth=3)
    ax5.plot(Vg_mw.columns,
             Vs_b_ctv.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3)
    ax5.plot(Vg_mw.columns,
             Vm_b_ctv.loc[stream_index,Vg_mw.columns]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g-', linewidth=3)
    ax5.set_title('Cumulative Sediment Accumulated on Channel Bed',fontsize=10)
    ax5.set_ylabel('Depth (m)',fontsize=10)
    ax5.set_xlabel('Year',fontsize=8)
    
    sum_transported_gravel=np.sum(Vg_t.loc[stream_index,Vg_mw.columns])
    sum_transported_sandb=np.sum(Vsb_t.loc[stream_index,Vg_mw.columns])
    sum_transported_sands=np.sum(Vss_t.loc[stream_index,Vg_mw.columns])
    sum_transported_mud=np.sum(Vm_t.loc[stream_index,Vg_mw.columns])    
    
    print('Stream Number', stream_index, ', Total gravel transported (10^6 m3)=',sum_transported_gravel/10**6)
    print('Stream Number', stream_index, ', Total bedload sand transported(10^6 m3)=',sum_transported_sandb/10**6)
    print('Stream Number', stream_index, ', Total suspended sand transported(10^6 m3)=',sum_transported_sands/10**6)
    print('Stream Number', stream_index, ', Total mud transported(10^6 m3)=',sum_transported_mud/10**6)
    print('Stream Number', stream_index, ', Total sediment transported (10^6 m3)=',
          (sum_transported_gravel+sum_transported_sandb+sum_transported_sands+sum_transported_mud)/10**6)
    print('--------------------')