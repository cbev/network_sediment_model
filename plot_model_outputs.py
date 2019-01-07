# -*- coding: utf-8 -*-
"""
Plot simulation resutls for Elwha Sediment Model: Landlab-DHSVM Coupling
Created on Wed Jun 14 12:34:06 2017

@author: Claire
"""
#%% Import modules and define functions
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import pickle
#%% INPUTS
# Directories
model_dir='C:/Users/Claire/Documents/GitHub/cbev_projects/elwha_network_model/'
model_data_folder=model_dir+'data/dhsvm_streamflow/output_32/'
results_directory=model_dir+'output/runs/dhsvm_streamflow/2018-10-22_1026/' ## CHANGE FOR EACH RUN!

# Set the streams and years to plot
streams_to_plot=[157, 267, 390] #[130, 144, 157, 166, 236, 250, 259, 267, 290, 309, 314, 358, 374, 382,385, 390]
#streams_to_plot=[199, 227, 280, 296, 308, 316]

# Daily review for last year
recycled_run='yes' #no

daily_start_date=datetime.date(2010,12,31) # for daily review- start the day before year of interest
daily_end_date=datetime.date(2011,12,31)# for daily review
evaluation_years_plot=np.arange(1927,2011) # for annual review - end year is one beyond your last year of interest

#daily_start_date=datetime.date(1950,12,31) # for daily review- start the day before year of interest
#daily_end_date=datetime.date(1951,12,31)# for daily review
#evaluation_years_plot=np.arange(1927, 1952) # for annual review - end year is one beyond your last year of interest

# Other general/ unchanging parameters
sg=2.65             # specific gravity 
bd_fine=1.13        # bulk density of fine sediments in reservoir 
bd_coarse=1.71      # bulk density of coarse sediments in reservoir 
rte=0.86            # Reservoir Trap Efficiency for Suspended Sediment
gage= 144 #[174, 153]
outlet=[103,106]
res_bed_streams=[103, 106, 112, 116, 121, 129, 135, 137, 144] #check with and without 144
n=len(evaluation_years_plot)
# Additional feeder streams: 104 (180m in res), 130 (951 m in res), 
# 144 (385m in res), 151  (1085m in res)

#%% Load network information 
os.chdir(model_data_folder)
network=pickle.load(open('network.py', 'rb'))
strm_orders=np.unique(network.segment_order)
streamflow=pickle.load(open('streamflow.py', 'rb'))
strm_link_vals=pickle.load(open('strm_link_vals.py', 'rb'))
n_strms=len(strm_link_vals) # number of streams/tributary areas

# Load and plot daily results
os.chdir(results_directory)
strm_orders=np.unique(network.segment_order)

Vg_mw=pd.DataFrame(np.load('Vg_mw.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_mw=pd.DataFrame(np.load('Vs_mw.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vm_mw=pd.DataFrame(np.load('Vm_mw.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]

Vg_b=pd.DataFrame(np.load('Vg_b.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_b=pd.DataFrame(np.load('Vs_b.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vm_b=pd.DataFrame(np.load('Vm_b.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]

Vg_b_ctv=pd.DataFrame(np.load('Vg_b_ctv.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_b_ctv=pd.DataFrame(np.load('Vs_b_ctv.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vm_b_ctv=pd.DataFrame(np.load('Vm_b_ctv.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume of sediment deposited in each stream after mass wasting event [m3]

Vg_cap=pd.DataFrame(np.load('Vg_cap.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume that stream has capacity to transport each day
Vsb_cap=pd.DataFrame(np.load('Vsb_cap.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date))  # Volume that stream has capacity to transport each day
Vss_cap=pd.DataFrame(np.load('Vss_cap.npy'), index=strm_link_vals,columns=pd.date_range(daily_start_date, daily_end_date)) # Volume that stream has capacity to transport each day

Vg_t=pd.DataFrame(np.load('Vg_t.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm
Vsb_t=pd.DataFrame(np.load('Vsb_t.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm
Vss_t=pd.DataFrame(np.load('Vss_t.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm
Vm_t=pd.DataFrame(np.load('Vm_t.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm

Ch_Fs=pd.DataFrame(np.load('Ch_Fs.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm
Ch_Fg=pd.DataFrame(np.load('Ch_Fg.npy'), index=strm_link_vals, columns=pd.date_range(daily_start_date, daily_end_date)) # Volume transported out of stream with each storm

LM_gage_rc=pd.DataFrame(np.load('LM_gage_rc.npy'), index=np.arange (0,len(streamflow.index)), columns=['g','sb','ss','m']) # Volume transported out of stream with each storm

#%%
for stream_index in streams_to_plot:  
    plot_title=''
    fig, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5,1, sharex=True, sharey=False)        
    
    ax1.plot(pd.date_range(daily_start_date, daily_end_date),
             Vg_mw.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=2, label='gravel')
    ax1.plot(pd.date_range(daily_start_date, daily_end_date),
             Vs_mw.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b--', linewidth=3, label='sand')
    ax1.plot(pd.date_range(daily_start_date, daily_end_date),
             Vm_mw.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g--', linewidth=3, label='mud')
    ax1.set_title('Stream Number '+str(stream_index)+'\nSediment Deposited from Landslide',fontsize=12)
    ax1.set_ylabel('Depth (m)',fontsize=8)
    ax1.legend()
    
    ax2.plot(pd.date_range(daily_start_date, daily_end_date), 
             Vg_cap.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=2)  
    ax2.plot(pd.date_range(daily_start_date, daily_end_date), 
             Vsb_cap.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b--', linewidth=2)
    ax2.set_title('Bedload Transport Capacity of Stream',fontsize=12)
    ax2.set_ylabel('Depth (m)',fontsize=8)
    #ax2.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax3.plot(pd.date_range(daily_start_date, daily_end_date),
             Vg_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--',linewidth=2)
    ax3.plot(pd.date_range(daily_start_date, daily_end_date),
             Vsb_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b--',linewidth=2)
    ax3.plot(pd.date_range(daily_start_date, daily_end_date),
             Vss_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'c--',linewidth=2)
    ax3.plot(pd.date_range(daily_start_date, daily_end_date),
             Vm_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g--',linewidth=2)
    ax3.set_title('Sediment Transported Out of Channel',fontsize=12)
    ax3.set_ylabel('Depth (m)',fontsize=8)
    #ax3.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax4.plot(pd.date_range(daily_start_date, daily_end_date),
             Vg_b.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=2)
    ax4.plot(pd.date_range(daily_start_date, daily_end_date),
             Vs_b.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b--', linewidth=2)
    ax4.plot(pd.date_range(daily_start_date, daily_end_date),
             Vm_b.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g--', linewidth=2)
    ax4.set_title('Sediment Accumulated on Channel Bed',fontsize=12)
    ax4.set_ylabel('Depth (m)',fontsize=8)
    
    ax5.plot(pd.date_range(daily_start_date, daily_end_date),
             Vg_b_ctv.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=2)
    ax5.plot(pd.date_range(daily_start_date, daily_end_date),
             Vs_b_ctv.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b--', linewidth=2)
    ax5.plot(pd.date_range(daily_start_date, daily_end_date),
             Vm_b_ctv.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g--', linewidth=2)
    ax5.set_title('Cumulative Sediment Accumulated on Channel Bed',fontsize=12)
    ax5.set_ylabel('Depth (m)',fontsize=12)
    ax5.set_xlabel('Year',fontsize=8)
    
    sum_transported_gravel=np.sum(Vg_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)])
    sum_transported_sandb=np.sum(Vsb_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)])
    sum_transported_sands=np.sum(Vss_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)])
    sum_transported_mud=np.sum(Vm_t.loc[stream_index,pd.date_range(daily_start_date, daily_end_date)])    
    
    print('Stream Number', stream_index, ', Total gravel transported (10^6 m3)=',sum_transported_gravel/10**6)
    print('Stream Number', stream_index, ', Total bedload sand transported(10^6 m3)=',sum_transported_sandb/10**6)
    print('Stream Number', stream_index, ', Total suspended sand transported(10^6 m3)=',sum_transported_sands/10**6)
    print('Stream Number', stream_index, ', Total mud transported(10^6 m3)=',sum_transported_mud/10**6)
    print('Stream Number', stream_index, ', Total sediment transported (10^6 m3)=',
          (sum_transported_gravel+sum_transported_sandb+sum_transported_sands+sum_transported_mud)/10**6)
    print('--------------------')

#%% Load and plot annual results
os.chdir(results_directory)
Vg_mw_annual=pd.DataFrame(np.load('Vg_mw_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_mw_annual=pd.DataFrame(np.load('Vs_mw_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vm_mw_annual=pd.DataFrame(np.load('Vm_mw_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]

Vg_b_annual=pd.DataFrame(np.load('Vg_b_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_b_annual=pd.DataFrame(np.load('Vs_b_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment depositedin in each stream after mass wasting event [m3]
Vm_b_annual=pd.DataFrame(np.load('Vm_b_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited  each stream after mass wasting event [m3]

Vg_b_ctv_annual=pd.DataFrame(np.load('Vg_b_ctv_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vs_b_ctv_annual=pd.DataFrame(np.load('Vs_b_ctv_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]
Vm_b_ctv_annual=pd.DataFrame(np.load('Vm_b_ctv_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume of sediment deposited in each stream after mass wasting event [m3]

Vg_cap_annual=pd.DataFrame(np.load('Vg_cap_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume that stream has capacity to transport each day
Vsb_cap_annual=pd.DataFrame(np.load('Vsb_cap_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume that stream has capacity to transport each day
Vss_cap_annual=pd.DataFrame(np.load('Vss_cap_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume that stream has capacity to transport each day

Vg_t_annual=pd.DataFrame(np.load('Vg_t_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume transported out of stream with each storm
Vsb_t_annual=pd.DataFrame(np.load('Vsb_t_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume transported out of stream with each storm
Vss_t_annual=pd.DataFrame(np.load('Vss_t_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume transported out of stream with each storm
Vm_t_annual=pd.DataFrame(np.load('Vm_t_annual.npy'), index=strm_link_vals, columns=np.unique(streamflow.index.year)) # Volume transported out of stream with each storm

for stream_index in streams_to_plot:  
    plot_title=''
    fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True, sharey=False)        
    
    ax1.plot(evaluation_years_plot,
             Vg_mw_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r-', linewidth=2)
    ax1.plot(evaluation_years_plot,
             Vs_mw_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=2)
    ax1.plot(evaluation_years_plot,
             Vm_mw_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'g--', linewidth=2)
    ax1.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
#    ax1.set_title('Stream Number '+str(stream_index)+'\nSediment Deposited from Mass Wasting, V${_d,_j}$',fontsize=10)
    ax1.set_title('Sediment Deposited from Mass Wasting, V${_d,_j}$',fontsize=10)
    ax1.set_ylabel('Depth\n(m)\n',fontsize=10)
    
#    # Depth
#    ax2.plot(evaluation_years_plot, 
#             Vg_cap_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'r-', linewidth=2)  
#    ax2.plot(evaluation_years_plot, 
#             Vsb_cap_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'b-', linewidth=2)
#    ax2.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
#    ax2.set_title('Bedload Transport Capacity of Stream, Total for Year',fontsize=10)
#    ax2.set_ylabel('Capacity\n(m2/day)',fontsize=10)

    # rate (m2/day)
    ax2.plot(evaluation_years_plot, 
             Vg_cap_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'r-', linewidth=2)  
    ax2.plot(evaluation_years_plot, 
             Vsb_cap_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'b-', linewidth=2)
    ax2.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
    ax2.set_title('Bedload Transport Capacity of Stream, V${_c,_j}$',fontsize=10)
    ax2.set_ylabel('Capacity\n(m2/day)\n',fontsize=10)
    
    # Depth    
#    ax3.plot(evaluation_years_plot,
#             Vg_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'r-',linewidth=2, label='Gravel, Bedload ')
#    ax3.plot(evaluation_years_plot,
#             Vsb_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'b-',linewidth=2, label='Sand, Bedload ')
#    ax3.plot(evaluation_years_plot,
#             Vss_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'c--',linewidth=2, label='Sand, Suspended Load')
#    ax3.plot(evaluation_years_plot,
#             Vm_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'g--',linewidth=2, label='Mud, Suspended Load')
#    ax3.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
#    ax3.set_title('Sediment Transported Out of Channel, Total for Year',fontsize=10)
#    ax3.set_ylabel('Capacity\n(m2/day)',fontsize=10)
    
    # rate (m2/day)
    ax3.plot(evaluation_years_plot,
             Vg_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'r-',linewidth=2, label='Gravel, Bedload ')
    ax3.plot(evaluation_years_plot,
             Vsb_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'b-',linewidth=2, label='Sand, Bedload ')
    ax3.plot(evaluation_years_plot,
             Vss_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'c--',linewidth=2, label='Sand, Suspended Load')
    ax3.plot(evaluation_years_plot,
             Vm_t_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*365.25),
             'g--',linewidth=2, label='Mud, Suspended Load')
    ax3.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
    ax3.set_title('Sediment Transported Out of Channel, V${_o,_j}$',fontsize=10)
    ax3.set_ylabel('Capacity\n(m2/day)\n',fontsize=10)
   
    if recycled_run=='no':
        ax4.plot(evaluation_years_plot,
                 (Vg_b_annual.loc[stream_index,evaluation_years_plot])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'r-', linewidth=2)
        ax4.plot(evaluation_years_plot,
                 (Vs_b_annual.loc[stream_index,evaluation_years_plot])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'b-', linewidth=2)
        ax4.plot(evaluation_years_plot,
                 (Vm_b_annual.loc[stream_index,evaluation_years_plot])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'g--', linewidth=2)
        ax4.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
        ax4.set_title('Sediment Stored on Channel Bed, V${_b,_j}$',fontsize=10)
        ax4.set_ylabel('Depth\n(m)\n',fontsize=10)
        
    if recycled_run=='yes':
        ax4.plot(evaluation_years_plot,
                 (Vg_b_annual.loc[stream_index,evaluation_years_plot[0]-1]+Vg_b_annual.loc[stream_index,evaluation_years_plot]-Vg_b_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'r-', linewidth=2)
        ax4.plot(evaluation_years_plot,
                 (Vs_b_annual.loc[stream_index,evaluation_years_plot[0]-1]+Vs_b_annual.loc[stream_index,evaluation_years_plot]-Vs_b_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'b-', linewidth=2)
        ax4.plot(evaluation_years_plot,
                 (Vm_b_annual.loc[stream_index,evaluation_years_plot[0]-1]+Vm_b_annual.loc[stream_index,evaluation_years_plot]-Vm_b_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
                 'g--', linewidth=2)
        ax4.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
        ax4.set_title('Sediment Stored on Channel Bed, V${_b,_j}$',fontsize=10)
        ax4.set_ylabel('Depth\n(m)\n',fontsize=10)
    
#    ax5.plot(evaluation_years_plot,
#             (Vg_b_ctv_annual.loc[stream_index,evaluation_years_plot]-Vg_b_ctv_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'r-', linewidth=2)
#    ax5.plot(evaluation_years_plot,
#             (Vs_b_ctv_annual.loc[stream_index,evaluation_years_plot]-Vs_b_ctv_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'b-', linewidth=2)
#    ax5.plot(evaluation_years_plot,
#             (Vm_b_ctv_annual.loc[stream_index,evaluation_years_plot]-Vm_b_ctv_annual.loc[stream_index,evaluation_years_plot[0]-1])/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
#             'g--', linewidth=2)
#    ax5.ticklabel_format(style='sci', axis='y',scilimits=(0.01,100))
#    ax5.set_title('Sediment Temporarily Stored on Channel Bed, Cumulative',fontsize=10)
#    ax5.set_ylabel('Depth\n(m)',fontsize=10)
#    ax5.set_xlabel('Year',fontsize=10)
    
    sum_transported_gravel=np.sum(Vg_t_annual.loc[stream_index,evaluation_years_plot])
    sum_transported_sandb=np.sum(Vsb_t_annual.loc[stream_index,evaluation_years_plot])
    sum_transported_sands=np.sum(Vss_t_annual.loc[stream_index,evaluation_years_plot])
    sum_transported_mud=np.sum(Vm_t_annual.loc[stream_index,evaluation_years_plot])    
    
    print('Stream Number', stream_index, ', Total gravel transported (10^6 m3)=',sum_transported_gravel/10**6)
    print('Stream Number', stream_index, ', Total bedload sand transported(10^6 m3)=',sum_transported_sandb/10**6)
    print('Stream Number', stream_index, ', Total suspended sand transported(10^6 m3)=',sum_transported_sands/10**6)
    print('Stream Number', stream_index, ', Total mud transported(10^6 m3)=',sum_transported_mud/10**6)
    print('Stream Number', stream_index, ', Total sediment transported (10^6 m3)=',
          (sum_transported_gravel+sum_transported_sandb+sum_transported_sands+sum_transported_mud)/10**6)
    print('--------------------')

#%% Cumulative reservoir sedimentation volume
ressum_trn_gravel=np.sum(np.sum(Vg_t_annual.loc[outlet,evaluation_years_plot], axis=0))
ressum_trn_sand_bed=np.sum(np.sum(Vsb_t_annual.loc[outlet,evaluation_years_plot], axis=0))
ressum_trn_sand_sus=rte*np.sum(np.sum(Vss_t_annual.loc[outlet,evaluation_years_plot], axis=0))
ressum_trn_mud=rte*np.sum(np.sum(Vm_t_annual.loc[outlet,evaluation_years_plot], axis=0))    

ressum_dep_gravel=np.sum(Vg_b_annual.loc[res_bed_streams,evaluation_years_plot[-1]]-Vg_b_annual.loc[res_bed_streams,evaluation_years_plot[0]-1])
ressum_dep_sand=np.sum(Vs_b_annual.loc[res_bed_streams,evaluation_years_plot[-1]]-Vs_b_annual.loc[res_bed_streams,evaluation_years_plot[0]-1])
ressum_dep_mud=np.sum(Vm_b_annual.loc[res_bed_streams,evaluation_years_plot[-1]]-Vm_b_annual.loc[res_bed_streams,evaluation_years_plot[0]-1])

gravel=(ressum_dep_gravel+ressum_trn_gravel)*sg/(bd_coarse*10**6)
sand=(ressum_dep_sand+ressum_trn_sand_bed+ressum_trn_sand_sus)*sg/(bd_coarse*10**6)
fine=(ressum_dep_mud+ressum_trn_mud)*sg/(bd_fine*10**6)

print('Total Reservoir Sedimentation (10^6 m3)=',gravel+sand+fine)
print('Gravel=', gravel)
print('Sand=', sand)
print('Fine=', fine)
print('Reservoir Percent Coarse=', 100*(gravel+sand)/(gravel+sand+fine))
print('Reservoir Percent Fines=', 100*fine/(gravel+sand+fine))

#%% Check results at Lake Mills gage
gagesum_transported_gravel=np.sum(np.sum(Vg_t_annual.loc[gage,evaluation_years_plot], axis=0))
gagesum_transported_sandb=np.sum(np.sum(Vsb_t_annual.loc[gage,evaluation_years_plot], axis=0))
gagesum_transported_sands=np.sum(np.sum(Vss_t_annual.loc[gage,evaluation_years_plot], axis=0))
gagesum_transported_mud=np.sum(np.sum(Vm_t_annual.loc[gage,evaluation_years_plot], axis=0)) 

bedload_sed=gagesum_transported_gravel+gagesum_transported_sandb
susp_sed=gagesum_transported_sands+gagesum_transported_mud

print('log10 (Bedload) (m2/day)- Daily Average=',
      np.log10(bedload_sed/(network.loc[gage,'width']*n*365.25)))
print('log10 (Suspended Load) (m2/day)- Daily Average',
      np.log10(susp_sed/(network.loc[gage,'width']*n*365.25)))
#%% Plot sedmient versus streamflow
w_const_ref=41.6
mod_dates=pd.date_range(start='1/1/'+str(evaluation_years_plot[0]), end='12/31/'+str(evaluation_years_plot[-1]))
LM_gage_rc.index=streamflow.index
LM_gage_rc_mod=LM_gage_rc.loc[mod_dates,:]

LM_gage_rc_q_m3d=streamflow.loc[mod_dates,gage]*3600*24
LM_gage_rc_qb_m3d=LM_gage_rc_mod.loc[:,['g','sb']].sum(axis=1)#/network.loc[gage,'width']
LM_gage_rc_qs_m3d=LM_gage_rc_mod.loc[:,['ss','m']].sum(axis=1)#/network.loc[gage,'width']

plt.figure()
plt.loglog(LM_gage_rc_q_m3d,LM_gage_rc_qb_m3d,'g.',markersize=0.5,label='bedload')
plt.loglog(LM_gage_rc_q_m3d,LM_gage_rc_qs_m3d,'r.',markersize=0.5,label='suspended load')
plt.xlabel('Q (m3/day)')
plt.ylabel('Sediment load (m3/day)')
plt.legend(loc='best')
#%% Rating curve of sediment and streamflow
LM_gage_rc_q_srt_m3d_srt=np.sort(LM_gage_rc_q_m3d)
LM_gage_rc_q_srt_m3d_argsort=np.argsort(LM_gage_rc_q_m3d)
LM_gage_rc_q_srt_m3d_ep=np.array(range(1,len(LM_gage_rc_q_srt_m3d_srt)+1))/(len(LM_gage_rc_q_srt_m3d_srt)+1)

LM_gage_rc_qb_m3d_qsrt=LM_gage_rc_qb_m3d[LM_gage_rc_q_srt_m3d_argsort]
LM_gage_rc_qs_m3d_qsrt=LM_gage_rc_qs_m3d[LM_gage_rc_q_srt_m3d_argsort]

w_const_ref=41.6
fig1, ax1=plt.subplots(1,1,figsize=(4,3))
ax2=ax1.twinx()
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qb_m3d_qsrt, 'g.',markersize=0.5, label='Bedload- Model')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qs_m3d_qsrt, 'b.',markersize=0.5, label='Suspended Load- Model')
ax2.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_q_srt_m3d_srt, 'k.',markersize=0.5, label='Streamflow')

ax1.set_xlabel('Fraction of time discharge equaled or exceeded', fontsize=12)
ax1.set_ylabel('Sediment Discharge (m$^{3}$/day)', fontsize=12)
ax2.set_ylabel('Streamflow Discharge (m$^{3}$/day)', fontsize=12)
ax1.grid(which='major', axis='x')
#ax1.set_ylim([10**-9, 10**3])
#ax2.set_ylim([10**-2, 10**1])
#plt.legend(loc='best')
#%%
## Comparison to observation-based rating curve
# Bedload Discharge (Curran et al., 2009- power equations from regression)
a_b=0.01       # regression coefficient from Curran et al., 2009
b_b=2.41       # regression coefficient from Curran et al., 2009
cf_b=1.13      # log-regression correction factor from Curran et al., 2009

#Suspended Load Concentration and Discharge (Curran et al., 2009- power equations from regression)
a_s=1.17*10**-4 # regression coefficient from Curran et al., 2009
b_s_c=3        # regression coefficient for SS concentration from Curran et al., 2009
b_s_l=4        # regression coefficient for SS load from Curran et al., 2009
cf_s=1.07      # log-regression correction factor from Curran et al., 2009
K=0.0864       # unit conversion factor from Curran et al., 2009
a_ps=175.42690003797424
b_ps=-0.21656296615646134
# Reservoir Trap Efficiency for Suspended Sediment
rte=0.86

# set constants and variables for the rating curve
Q_RC=LM_gage_rc_q_srt_m3d_srt/(24*3600)
dp=LM_gage_rc_q_srt_m3d_ep[2]-LM_gage_rc_q_srt_m3d_ep[1]
n_years_GC=len(evaluation_years_plot)

rho_w=1000
g=9.81
constant_width=41.6
a_Uf=0.10
b_Uf=0.58
a_Df=0.24
b_Df=0.41
a_n=1.08
b_n=-0.44
ng_obs_bar=0.02417
S=0.005
Ch_Fs=0.37
Ch_d90_m=0.0275 # m -  d90 of channel bed= 27.5 mm=0.0275 m
Ch_Dsand_m=0.00093 # m- d_sand of channel bed= 0.93 mm = 0.00093 m
Ch_Dgravel_m=0.0132 # m- d_gravel of channel bed= 13.2 mm = 0.0132 m
Ch_Dmean_m=0.0125 # m- d_mean of channel bed= 12.5 mm = 0.0125 m

def compute_channel_geom(constant_width, a_Uf, b_Uf,a_Df, b_Df, a_n, b_n,
                         ng_obs_bar, Q, S): # a_Wf, b_Wf
#    W=a_Wf*Q**b_Wf
    W=constant_width*np.ones(len(Q))
    U=a_Uf*Q**b_Uf
    D=a_Df*Q**b_Df
    Rh=D
    ng=ng_obs_bar*(a_n*D**b_n)
    tau=rho_w*g*((ng*U)**(3/2))*S**(1/4)

    return W, U, D, Rh, tau 

def run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_BL_obs):
# Two-fraction Equation
# Constants
    sg=2.65
    rho_w=1000
    Ch_Fg=1-Ch_Fs
    
    A_gravel_W2=14
    chi_gravel_W2=0.894
    exp_gravel_W2=0.5
    phi_prime_gravel_W2=1.35
    
    A_sand_W2=14
    chi_sand_W2=0.894
    exp_sand_W2=0.5
    phi_prime_sand_W2=1.35
    
    tau_star_rsm_W2=0.021+0.015*np.exp(-20*Ch_Fs) # dimensionless reference shear stress for mean grain size
    tau_rsm_W2=tau_star_rsm_W2*(sg-1)*rho_w*g*Ch_Dmean_m # reference shear stress for mean grain size [N/m2]
    b_sand_W2=0.67/(1+np.exp(1.5-(Ch_Dsand_m/Ch_Dmean_m))) # b parameter for sand
    b_gravel_W2=0.67/(1+np.exp(1.5-(Ch_Dgravel_m/Ch_Dmean_m))) # b parameter for gravel
    tau_r_sand_W2=tau_rsm_W2*(Ch_Dsand_m/Ch_Dmean_m)**b_sand_W2 # reference tau for sand [N/m2]
    tau_r_gravel_W2=tau_rsm_W2*(Ch_Dgravel_m/Ch_Dmean_m)**b_gravel_W2 # reference tau for gravel [N/m2]
        
    phi2_line=np.arange(0.1,20,0.1)
    Wstar2_line=np.empty(len(phi2_line))
    for i in np.arange (0,(len(phi2_line))):
        if phi2_line[i] < phi_prime_gravel_W2:
            Wstar2_line[i]=0.002*(phi2_line[i])**7.5
        elif chi_gravel_W2/((phi2_line[i])**exp_gravel_W2)>1: # Check that term in paraentheses is not negative
            Wstar2_line[i]=0
        else:
            Wstar2_line[i]=A_gravel_W2*((1-(chi_gravel_W2/((phi2_line[i])**exp_gravel_W2)))**(4.5))
    
    # Modeled Samples
    phi_gravel_W2=tau_BL_obs/tau_r_gravel_W2
    phi_sand_W2=tau_BL_obs/tau_r_sand_W2
    
    Wstar_gravel_W2=np.empty([len(phi_gravel_W2)])
    Wstar_sand_W2=np.empty([len(phi_gravel_W2)])
    dis_m2s_gravel_W2=np.empty([len(phi_gravel_W2)])
    dis_m2s_sand_W2=np.empty([len(phi_gravel_W2)])
    dis_m2s_total_W2=np.empty([len(phi_gravel_W2)])
    
    for i in np.arange(0, len(tau_BL_obs)):
        u_star=(tau_BL_obs[i]/rho_w)**0.5
    
        # Gravel
        if phi_gravel_W2[i]<phi_prime_gravel_W2:
            Wstar_gravel_W2[i]=0.002*(phi_gravel_W2[i])**7.5
        elif chi_gravel_W2/((phi_gravel_W2[i])**exp_gravel_W2)>1: # Checka that term in paraentheses is not negative
           Wstar_gravel_W2[i]=0
        else:
            Wstar_gravel_W2[i]=A_gravel_W2*((1-(chi_gravel_W2/((phi_gravel_W2[i])**exp_gravel_W2)))**(4.5))
        # Sand
        if phi_sand_W2[i]<phi_prime_sand_W2:
            Wstar_sand_W2[i]=0.002*(phi_sand_W2[i])**7.5
        elif chi_sand_W2/((phi_sand_W2[i])**exp_sand_W2)>1:
           Wstar_sand_W2[i]=0
        else:
            Wstar_sand_W2[i]=A_sand_W2*((1-(chi_sand_W2/((phi_sand_W2[i])**exp_sand_W2)))**(4.5))
        
        # Discharge in m2/s
        dis_m2s_gravel_W2[i]=((u_star)**3)*Wstar_gravel_W2[i]*Ch_Fg/((sg-1)*g)
        dis_m2s_sand_W2[i]=((u_star)**3)*Wstar_sand_W2[i]*Ch_Fs/((sg-1)*g)
        dis_m2s_total_W2[i]=dis_m2s_gravel_W2[i]+dis_m2s_sand_W2[i]

    return (phi2_line, Wstar2_line, phi_gravel_W2, phi_sand_W2, 
            Wstar_gravel_W2, Wstar_sand_W2, dis_m2s_gravel_W2, dis_m2s_sand_W2, 
            dis_m2s_total_W2)  
    
    
def compute_SS_RC (a_s, b_s_c, b_s_l, cf_s, K, Q, W):
    sg=2.65
    dis_MgD=cf_s*a_s*K*Q**b_s_l #Mg/D
    dis_ss_m3s=dis_MgD/(sg*24*3600)
    dis_ss_m2s= dis_ss_m3s/W
    return (dis_ss_m2s, dis_ss_m3s)

# Suspended Sediment
Q_s_percentile=cf_s*a_s*K*Q_RC**b_s_l    # Suspended load (Mg/d) for each percentile of interest at LM
ps_RC=np.empty(len(Q_RC))                # percent silt for each percentile of interest at LM
for i in range (0, len(Q_RC)):
    if Q_RC[i]<30:
        ps_RC[i]=0.84
    elif (Q_RC[i]>=30) and (Q_RC[i]<150):
        ps_RC[i]=(a_ps*Q_RC[i]**b_ps)/100
    else: ps_RC[i]=0.59 
Q_s_ann_silt=ps_RC*Q_s_percentile*dp*365.25     # Annual contribution of SS silt (Mg/yr) for each percentile of interest at LM
Q_s_ann_sand=(1-ps_RC)*Q_s_percentile*dp*365.25 # Annual contribution of SS sand (Mg/yr) for each percentile of interest at LM
Q_s_totann_silt=np.sum([Q_s_ann_silt])  # Annual SS silt contribution (Mg/yr) at LM
Q_s_totann_sand=np.sum([Q_s_ann_sand])  # Annual SS sand contribution (Mg/yr) at 
Q_s_totann=Q_s_totann_sand+Q_s_totann_silt
Q_s_Vsed_totann_LM=(Q_s_totann_silt+Q_s_totann_sand)/sg # Annual SS passing Lake Mills gage (m3/yr)
# Q_s_erosion=1000*Q_s_Vsed_totann_LM/A_LM     # Annual ss eroded from hillslopes (mm/year)

## Distinguish between silt and sand in reservoir
Q_s_Vsed_totann_silt_GC=Q_s_totann_silt/bd_fine # Annual silt in reservoir- before being transported out
Q_s_Vsed_totann_sand_GC=Q_s_totann_sand/bd_coarse # Annual sand in reservoir
Q_s_Vsed_totann_GC=Q_s_Vsed_totann_silt_GC+Q_s_Vsed_totann_sand_GC# Annual SS contribution at Glines Canyon dam (m3/yr)
## Incorporate reservir trap efficiency, assuming only silt flows over (not sand)
Q_s_Vsed_totann_silt_GC=Q_s_Vsed_totann_silt_GC-Q_s_Vsed_totann_GC*(1-rte)
Q_s_Vsed_totann_GC=Q_s_Vsed_totann_silt_GC+Q_s_Vsed_totann_sand_GC

Q_s_sand_Vsed_total_GC=Q_s_Vsed_totann_sand_GC*n_years_GC
Q_s_silt_Vsed_total_GC=Q_s_Vsed_totann_silt_GC*n_years_GC
Q_s_Vsed_total_GC=Q_s_Vsed_totann_GC*n_years_GC


# Bedload
Q_b_percentile=cf_b*a_b*Q_RC**b_b             # Bedload (Mg/d) for each percentile of interest at LM
Q_b_annual_contribution=Q_b_percentile*dp*365.25      # Annual contribution of bedload (Mg/yr) for each percentile of interest at LM
Q_b_annual_total=np.sum([Q_b_annual_contribution])    # Annual bedload contribution (Mg/yr) at LM
Q_b_Vsed_annual_total_LM=Q_b_annual_total/sg          # Annual bedload passing Lake Mills gage (m3/yr)- since using specific gravity
# Q_b_erosion=1000*Q_b_Vsed_annual_total_LM/A_LM        # Annual bedload eroded from watershed (mm/year)
Q_b_Vsed_annual_total_GC=Q_b_annual_total/bd_coarse   # Annual bedload contribution at Glines Canyon dam (m3/yr)- since using bulk density
Q_b_Vsed_total_GC=Q_b_Vsed_annual_total_GC*n_years_GC # Total bedload at Glines Canyon Dam (m3)

# Computed dam sedimentation rate, incorporating trap efficiency of reservoir with suspended sediment
Q_t_Vsed_annual_total_GC=Q_s_Vsed_totann_GC+Q_b_Vsed_annual_total_GC  # total sedmentation rate per year
Q_t_Vsed_lifetime_total=Q_t_Vsed_annual_total_GC*n_years_GC        # Estimate sediment accumulation over dam lifetime (m3)
Q_b_Vsed_pcnt_GC=Q_b_Vsed_annual_total_GC/Q_t_Vsed_annual_total_GC # percent of bedload in dam  (m3)
Q_s_Vsed_pcnt_GC=Q_s_Vsed_totann_GC/Q_t_Vsed_annual_total_GC        # percent of SS in dam (m3)

#Q_t_lifetime_dif=Q_t_Vsed_lifetime_total-Vsed_GC    # Difference between modeled and estimated sediment accumulation (m3)
#Q_t_lifetime_pctdif=100*Q_t_lifetime_dif/Vsed_GC    # percent difference

# Rating Curves
# compute channel geometry
W_RC, U_RC, D_RC, Rh_RC, tau_RC=compute_channel_geom(constant_width,a_Uf, b_Uf,a_Df, b_Df, a_n, b_n, ng_obs_bar, Q_RC, S) 

# Suspended sediment for plot
dis_ss_m2s_RC, dis_ss_m3s_RC=compute_SS_RC (a_s, b_s_c, b_s_l, cf_s, K, Q_RC, W_RC)

# Empirical Bedload for plot
dis_m3s_bedload_RC=Q_b_percentile*1000/(24*3600*sg*rho_w) # convert percentile discharge to m3/s

# Empirical Suspended Sediment for plot
dis_m3s_suspended_RC=Q_s_percentile*1000/(24*3600*sg*rho_w) # convert percentile discharge to m3/s

# Wilcock and Crowe two-fraction for rating curve
phi_line_RC, Wstar_line_RC, phi_gravel_RC, phi_sand_RC, Wstar_gravel_RC, Wstar_sand_RC,dis_m2s_gravel_RC, dis_m2s_sand_RC, dis_m2s_total_RC=run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_RC)
# Wilcock and Crowe two-fraction- Annual contribution of sediment (m3/yr) at LM
WC_ann_dis_m3yr=np.nansum([dis_m2s_total_RC*W_RC*3600*24*dp*365.25])
WC_ann_gravel_m3yr=np.nansum([dis_m2s_gravel_RC*W_RC*3600*24*dp*365.25])
WC_ann_sand_m3yr=np.nansum([dis_m2s_sand_RC*W_RC*3600*24*dp*365.25])
# Wilcock and Crowe two-fraction- Annual reservoir sedimentation (m3/yr) 
WC_ann_res_m3yr=WC_ann_dis_m3yr*sg/bd_coarse
WC_gravel_ann_res_m3yr=WC_ann_gravel_m3yr*sg/bd_coarse
WC_sand_ann_res_m3yr=WC_ann_sand_m3yr*sg/bd_coarse

n_bray_pre=(1/(np.sqrt(8*g)))*(Rh_RC**(1/6))/(1.26-2.16*np.log10(Ch_d90_m/np.float64(Rh_RC)))  
tau_pre=rho_w*g*((n_bray_pre*U_RC)**(3/2))*S**(1/4)
# Wilcock and Crowe two-fraction for rating curve
phi_line_RC_pre, Wstar_line_RC_pre, phi_gravel_RC_pre, phi_sand_RC_pre, Wstar_gravel_RC_pre, Wstar_sand_RC_pre, dis_m2s_gravel_RC_pre, dis_m2s_sand_RC_pre, dis_m2s_total_RC_pre=run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_pre)
# Wilcock and Crowe two-fraction- Annual contribution of sediment (m3/yr) at LM
WC_ann_dis_m3yr_pre=np.nansum([dis_m2s_total_RC_pre*W_RC*3600*24*dp*365.25])
WC_ann_gravel_m3yr_pre=np.nansum([dis_m2s_gravel_RC_pre*W_RC*3600*24*dp*365.25])
WC_ann_sand_m3yr_pre=np.nansum([dis_m2s_sand_RC_pre*W_RC*3600*24*dp*365.25])
# Wilcock and Crowe two-fraction- Annual reservoir sedimentation (m3/yr) 
WC_ann_res_m3yr_pre=WC_ann_dis_m3yr_pre*sg/bd_coarse
WC_gravel_ann_res_m3yr_pre=WC_ann_gravel_m3yr_pre*sg/bd_coarse
WC_sand_ann_res_m3yr_pre=WC_ann_sand_m3yr_pre*sg/bd_coarse

fig1, ax1=plt.subplots(1,1,figsize=(4,3))
ax2=ax1.twinx()
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qb_m3d_qsrt, 'g.',markersize=0.5, label='Bedload- Model')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qs_m3d_qsrt, 'c.',markersize=0.5, label='Suspended Load- Model')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m2s_total_RC*3600*24*W_RC,'b.',markersize=2, label='Bedload - Rating Curve (W&C-2F, ng,E)')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m3s_suspended_RC*3600*24,'r.',markersize=2, label='Suspended Load- Rating Curve')
ax2.semilogy(1-LM_gage_rc_q_srt_m3d_ep, Q_RC*3600*24,'k.',markersize=2, label='Streamflow- Model')

ax1.set_xlabel('Fraction of time discharge equaled or exceeded', fontsize=12)
ax1.set_ylabel('Sediment Discharge (m$^{3}$/day)', fontsize=12)
ax2.set_ylabel('Streamflow Discharge (m$^{3}$/day)', fontsize=12)
ax1.grid(which='major', axis='x')
#ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m3s_bedload_RC*3600*24,'g.',markersize=3, label='Bedload - USGS')
#ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m2s_total_RC_pre*3600*24*W_RC,'c.',markersize=3, label='Bedload - W&C-2F, ng,b')

#LM_gage_rc_qb_m2d_qsrt=np.sort(LM_gage_rc_qb_m3d)
#LM_gage_rc_qs_m2d_qsrt=np.sort(LM_gage_rc_qs_m3d)
#
#w_const_ref=41.6
#fig1, ax1=plt.subplots(1,1,figsize=(8,6))
#ax2=ax1.twinx()
#ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qb_m3d_qsrt, 'g.',markersize=0.5, label='bedload')
#ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qs_m3d_qsrt, 'b.',markersize=0.5, label='suspended load')
#ax2.semilogy(1-LM_gage_rc_q_srt_m3d_ep,(LM_gage_rc_q_srt_m3d_srt/w_const_ref*3600*24), 'k.',markersize=0.5, label='streamflow')
#%%
fig1, ax1=plt.subplots(1,1,figsize=(4,3))
ax2=ax1.twinx()
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qb_m3d_qsrt, 'g.',markersize=6, label='Bedload- Model')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep,LM_gage_rc_qs_m3d_qsrt, 'c.',markersize=6, label='Suspended Load- Model')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m2s_total_RC*3600*24*W_RC,'b.',markersize=6, label='Bedload - Rating Curve\n(W&C-2F, ng,E)')
ax1.semilogy(1-LM_gage_rc_q_srt_m3d_ep, dis_m3s_suspended_RC*3600*24,'r.',markersize=6, label='Suspended Load- Rating Curve\n(Regression-based)')
ax2.semilogy(1-LM_gage_rc_q_srt_m3d_ep, Q_RC*3600*24,'k.',markersize=6, label='Streamflow- Model')

ax1.set_xlabel('Fraction of time discharge equaled or exceeded', fontsize=12)
ax1.set_ylabel('Sediment Discharge (m$^{3}$/day)', fontsize=12)
ax2.set_ylabel('Streamflow Discharge (m$^{3}$/day)', fontsize=12)
#ax1.grid(which='major', axis='x')

ax1.legend(loc='best')
ax2.legend(loc='best')

#%% Annual reservoir sedimentation volume
ressum_trn_gravel_an=np.sum(Vg_t_annual.loc[outlet,evaluation_years_plot], axis=0)
ressum_trn_sand_bed_an=np.sum(Vsb_t_annual.loc[outlet,evaluation_years_plot], axis=0)
ressum_trn_sand_sus_an=rte*np.sum(Vss_t_annual.loc[outlet,evaluation_years_plot], axis=0)
ressum_trn_mud_an=rte*np.sum(Vm_t_annual.loc[outlet,evaluation_years_plot], axis=0) 

#Vg_b_annual.loc[res_bed_streams,evaluation_years_plot[0]-1].sum()

ressum_dep_gravel_an=Vg_b_annual.loc[res_bed_streams,evaluation_years_plot].sum()-Vg_b_annual.loc[res_bed_streams,evaluation_years_plot-1].sum().values
ressum_dep_sand_an=Vs_b_annual.loc[res_bed_streams,evaluation_years_plot].sum()-Vs_b_annual.loc[res_bed_streams,evaluation_years_plot-1].sum().values
ressum_dep_mud_an=Vm_b_annual.loc[res_bed_streams,evaluation_years_plot].sum()-Vm_b_annual.loc[res_bed_streams,evaluation_years_plot-1].sum().values

gravel_res_ann=(ressum_dep_gravel_an+ressum_trn_gravel_an)*sg/(bd_coarse)
sand_res_ann=(ressum_dep_sand_an+ressum_trn_sand_bed_an+ressum_trn_sand_sus_an)*sg/(bd_coarse)
fine_res_ann=(ressum_dep_mud_an+ressum_trn_mud_an)*sg/(bd_fine)

for i in gravel_res_ann.index:
    if gravel_res_ann[i]<0:
        gravel_res_ann[i]=0

for i in sand_res_ann.index:
    if sand_res_ann[i]<0:
        sand_res_ann[i]=0
        
for i in fine_res_ann.index:
    if fine_res_ann[i]<0:
        fine_res_ann[i]=0

total_ann=gravel_res_ann+sand_res_ann+fine_res_ann
#%%
(2.5*10**6)*365.25/1000

#%% Annual reservoir sedmientation
streamflow_res_ann=streamflow.loc[mod_dates,gage].groupby(pd.Grouper(freq="A")).sum()*3600*24
fig1, (ax1, ax2)=plt.subplots(2,1,figsize=(8,4))
ax3=ax2.twinx()
ax1.plot(evaluation_years_plot, fine_res_ann/1000,'g',label='Mud')
ax1.plot(evaluation_years_plot, sand_res_ann/1000,'b',label='Sand')
ax1.plot(evaluation_years_plot, gravel_res_ann/1000,'r',label='Gravel')
ax1.set_ylabel('Sedimentation\nVolume (10$^{3}$ m$^{3}$/yr)',fontsize=12)

ax2.plot(evaluation_years_plot, total_ann/1000,'k--',label='Total Sediment')
ax2.set_ylabel('Sedimentation\nVolume (10$^{3}$ m$^{3}$/yr)',fontsize=12)
ax3.plot(evaluation_years_plot, streamflow_res_ann/(10**6),'k', label='Streamflow')
ax3.set_ylabel('Streamflow\nVolume (10$^{6}$ m$^{3}$/yr)',fontsize=12)

#ax3.legend(loc='upper right')
#ax1.legend(loc='upper center', ncol=3)
#ax2.legend(loc='upper center')


#%% Annual reservoir sedmientation
streamflow_res_ann=streamflow.loc[mod_dates,gage].groupby(pd.Grouper(freq="A")).sum()*3600*24
fig1, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))
ax3=ax2.twinx()
ax1.plot(evaluation_years_plot, ressum_trn_mud_an,'g--',label='Mud')
ax1.plot(evaluation_years_plot, ressum_trn_sand_sus_an,'c--',label='Sand- suspended')
ax1.plot(evaluation_years_plot, ressum_trn_sand_bed_an,'b--',label='Sand- bedload')
ax1.plot(evaluation_years_plot, ressum_trn_gravel_an,'r--',label='Gravel')

ax1.plot(evaluation_years_plot, ressum_dep_mud_an,'g',label='Mud')
ax1.plot(evaluation_years_plot, ressum_dep_sand_an,'b',label='Sand')
ax1.plot(evaluation_years_plot, ressum_dep_gravel_an,'r',label='Gravel')
ax2.legend(loc='upper center')
ax2.legend(loc='upper left')
ax3.legend(loc='upper right')

#%% Cumulative reservoir seidmentation 
streamflow_res_ann=streamflow.loc[mod_dates,gage].groupby(pd.Grouper(freq="A")).sum()
fig1, (ax1, ax2, ax3)=plt.subplots(3,1,figsize=(10,6))
ax1.plot(evaluation_years_plot, fine_res_ann.cumsum(),'g',label='Mud')
ax1.plot(evaluation_years_plot, sand_res_ann.cumsum(),'b',label='Sand')
ax1.plot(evaluation_years_plot, gravel_res_ann.cumsum(),'r',label='Gravel')
ax3.plot(evaluation_years_plot, gravel_res_ann.cumsum()+sand_res_ann.cumsum()+fine_res_ann.cumsum(),'c',label='Total Sediment')
ax2.plot(evaluation_years_plot, streamflow_res_ann.cumsum(),'k', label='Streamflow')
#%%
plt.figure()
plt.plot(evaluation_years_plot, Vs_b_annual.loc[res_bed_streams,evaluation_years_plot].sum(),'k')
plt.plot(evaluation_years_plot, Vs_b_annual.loc[res_bed_streams,evaluation_years_plot-1].sum().values,'r')

#plt.plot(evaluation_years_plot,sand_res_ann,'b')
#%%
plt.plot(evaluation_years_plot, ressum_dep_gravel_an,'k')
plt.plot(evaluation_years_plot,gravel_res_ann,'b')
#%% Check mass balance of an individual link
### INPUTS ###
check_stream=350
check_year=1923
##############
check_feeders=network.index[np.where(network['dest_channel_id']==check_stream)[0]]
#Vg_b_annual.loc[check_stream,check_year-1]-Vg_b_annual.loc[check_stream,check_year]+np.sum(Vg_t_annual.loc[check_feeders,check_year])-Vg_t_annual.loc[check_stream,check_year]+Vg_mw_annual.loc[check_stream,check_year]

Vs_b_annual.loc[check_stream,check_year-1]-Vs_b_annual.loc[check_stream,check_year]+np.sum(Vss_t_annual.loc[check_feeders,check_year])-Vss_t_annual.loc[check_stream,check_year]+np.sum(Vsb_t_annual.loc[check_feeders,check_year])-Vsb_t_annual.loc[check_stream,check_year]+Vs_mw_annual.loc[check_stream,check_year]


#%% test
stream_index=103
Vg_b_ctv_annual.loc[stream_index,evaluation_years_plot[0]-1]

