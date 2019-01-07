"""
Elwha Data Analysis- Lake Mills Gage
Hydraulic Geometry and Sediment Transport Modeling
Written by Claire Beveridge, Univeristy of Washington

This notebook uses streamflow, sediment, and channel geometry data collected at 
the USGS gage 12044900- Elwha River Above Lake Mills Near Port Angeles, WA. 
Streamflow measurements from the USGS online portal and Childers et al. (2000) 
study are used to develop hydraulic geometry relationships at the Lake Mills 
gage. These hydraulic relationships are then used to parameterize the Wilcock
and Crowe (2003) equation for bedload transport. Estimates of reservoir 
sedimentation for the lifespan of the Glines Canyon Dam (84 years) using the 
Wilcock and Crowe (2003) bedload equations along with the bedload rating curve
 equation developed by Curran et al. (2009) are made using the daily streamflow 
rating curve for gage.  Suspended sediment accumulation is also estimated using 
the suspended sediment rating curve parameterization developed by Curran et al.
(2009).
"""
#%% Decisions
width='constant' # constant, regression
constant_width=41.6 # if width is constant, enter the constant value
#%%
# 1. SCRIPT SETUP AND PREPARATION- VARIABLES, FUNCTIONS, ETC.
# Directories
homedir='C:/Users/Claire/Documents/GitHub/cbev_projects/elwha_sediment/' # location of script
datadir='D:/GoogleDrive/Elwha/Data/' # location of Mills_AllData_Merged

# Modules
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
from scipy.interpolate import interp1d
import scipy.stats
from scipy.stats import t
import pandas as pd
import pickle

##############################################################################
# Parameters, Constants, etc.
g=9.81              # acceleration of gravity (m/s2)
rho_w=1000          # density of water (kg/m3)
sg=2.65             # specific gravity of coarse sediments (unitless)
tau_star_c=0.055    # 0.045 or 0.0495 (Wong and Parker)   # Critical dimensionless shear stress for large sands/gravels
S=0.005             # Slope
A_LM=513198000      # Lake Mills drainage area (m2) [In GIS, this is 512958656 m2 per TauDEM d-infinity]
bd_fine=1.13        # bulk density of fine sediments
bd_coarse=1.71      # bulk density of coarse sediments
n_years_GC=84       # number of years sed. accumualted behind Glines Canyon dam- assume for 10/1/1926-9/31/2010 
v=1*10**-6          # kinmatic viscosity
Vsed_GC=16100000    # Estimate of ~16.1 million (+/- 2.4 million) m3 for Glines Canyon dam (GC) (Randle et al., 2015)                        

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

# Reservoir Trap Efficiency for Suspended Sediment
rte=0.86

# Date windows for the rating curve
FDC_1_st_date=datetime.date(1994, 10, 1)
FDC_1_ed_date=datetime.date(1997, 9, 30)
FDC_2_st_date=datetime.date(2004, 10, 1)
FDC_2_ed_date=datetime.date(2011, 9, 30)
##############################################################################

#%%
# Functions 
def compute_NSE_rs (modeled, observed):
    NSE=1-((np.sum((modeled-observed)**2))/(np.sum((modeled-np.mean(observed))**2)))
    WC,WC_0,WC_r, WC_p, WC_err=scipy.stats.linregress(modeled, observed)
    r2=WC_r**2
    print('r2=',r2)
    print('NSE=',NSE)
    return NSE, r2
       
def find_nearest(array,value):
    '''
    Input a value and numpy array. Returns the index of the array entry that is 
    closest the the input value.
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def lin_fit(x,y):
    '''
    Source: https://github.com/KirstieJane/STATISTICS/blob/master/CIs_LinearRegression.py
    Predicts the values for a best fit between numpy arrays x and y
    
    Parameters
    ----------
    x: 1D numpy array
    y: 1D numpy array (same length as x)
    
    Returns
    -------
    p:     parameters for linear fit of x to y
    y_err: 1D array of difference between y and fit values    
               (same length as x)
    
    '''
    
    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    y_err = y - fit
    
    return p, y_err # p is function for polynomial, y_error is error

def conf_calc(x, y_err, c_limit):
    '''
    Source: https://github.com/KirstieJane/STATISTICS/blob/master/CIs_LinearRegression.py
    Calculates confidence interval of regression between x and y
    
    Parameters
    ----------
    x:       1D numpy array
    y_err:   1D numpy array of residuals (y - fit)
    c_limit: (optional) float number representing the area to the left
             of the critical value in the t-statistic table
             eg: for a 2 tailed 95% confidence interval (the default)
                    c_limit = 0.975
    Returns
    -------
    confs: 1D numpy array of predicted y values for x inputs
    
    '''
    # Define the variables you need
    # to calculate the confidence interval
    mean_x = np.mean(x)			# mean of x
    n = len(x)				# number of samples in origional fit
    tstat = t.ppf(c_limit, n-1)         # appropriate t value
    s_err = np.sum(np.power(y_err,2))	# sum of the squares of the residuals

    # create series of new test x-values to predict for
    p_x = np.linspace(np.min(x),np.max(x),50)

    confs = tstat * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
			((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))

    return p_x, confs
    
# ----------------------------------------------------------------------------

def ylines_calc(p, p_x, confs):
    '''
    Source: https://github.com/KirstieJane/STATISTICS/blob/master/CIs_LinearRegression.py
    Calculates the three lines that will be plotted
    
    Parameters
    ----------
    p_x:   1D array with values spread evenly between min(x) and max(x)
    confs: 1D array with confidence values for each value of p_x  
    
    Returns
    -------
    p_y:    1D array with values corresponding to fit line (for p_x values)
    upper:  1D array, values corresponding to upper confidence limit line
    lower:  1D array, values corresponding to lower confidence limit line
    
    '''
    # now predict y based on test x-values
    p_y = p(p_x)
    
    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    return p_y, lower, upper
   
# ----------------------------------------------------------------------------

def compute_ng (d_n, Rh_n, Ch_d50_m, Ch_d65_m, Ch_d84_m, Ch_d90_m, 
                ng_df_index):
    '''
    Parameters
    ----------
    ng_df_index: parameter which goes in the index (e.g., Q, H,...)
    d_n: set of channel depth measurements
    Rh_n: set of hydraulic radius measurements
    Ch_d50_m, Ch_d65_m, Ch_d84_m, Ch_d90_m: d50, d65, d84, d90 in [m]
    
    Returns
    -------
    ng_df: data frame with different calculations for n, organized based on 
    the input parameter
    '''
    n_array_names=['Ferguson, 2007',  'Leopold & Wolman, 1957','Hey, 1979',
               'Limerinos, 1970', 'Bray, 1979', 'Bathurst, 1985',
               'Griffiths, 1981', 'Smart & Jaggi, 1983', 'Strickler, 1923',
               'Keulegan, 1938']
    # Ferguson 2007
    a1=6.5
    a2=2.5
    d_ratio=(d_n/Ch_d84_m) 
    ferguson_8f=(a1*a2*d_ratio)/(np.sqrt((a1**2)+((a2**2)*(d_ratio)**(5/3))))
    n_ferguson=(1/ferguson_8f)*(Rh_n**(1/6))/np.sqrt(g)
    # Leopold and Wolman (1957)
    n_leopold=(1/(np.sqrt(8*g)))*(Rh_n**(1/6))/(1-2*np.log10(Ch_d84_m/Rh_n))    
    # Hey 1979
    n_hey=(1/(np.sqrt(8*g)))*(Rh_n**(1/6))/(1.02-2.03*np.log10(Ch_d84_m/Rh_n))   
    # Limerinos (1970)
    n_limerinos=(1/(np.sqrt(8*g)))*(Rh_n**(1/6))/(1.16-2*np.log10(Ch_d84_m/Rh_n))    
    # Bray (1982)
    n_bray=(1/(np.sqrt(8*g)))*(Rh_n**(1/6))/(1.26-2.16*np.log10(Ch_d90_m/Rh_n))    
    # Bathurst 1985
    n_bathurst=(1/np.sqrt(8*g))*(Rh_n**(1/6))/(1.42-1.99*np.log10(Ch_d84_m/Rh_n))    
    # Griffiths 1981
    n_griffiths=(1/(np.sqrt(8*g)))*(Rh_n**(1/6))/(0.76-1.98*np.log10(Ch_d50_m/Rh_n))   
    # Smart and Jaggi 1983
    Z90=(Rh_n/Ch_d90_m) 
    smart_8f=5.75*((1-(np.exp(-0.05*Z90/(S**0.5))))**0.5)*np.log10(8.2*Z90)
    n_smart=(1/smart_8f)*(Rh_n**(1/6))/np.sqrt(g)   
    # Manning-Strickler
    n_strickler=0.0475*Ch_d50_m**(1/6)    
    # Keulegan 1938
    n_keulegan=(1/np.sqrt(8*g))*(Rh_n**(1/6))/(2.21-2.04*np.log10(Ch_d50_m/Rh_n))
    # Create data frame
    ng_df=pd.DataFrame(data=[n_ferguson, n_leopold, n_hey, n_limerinos, 
                        n_bray, n_bathurst, n_griffiths, n_smart, 
                        np.repeat(n_strickler, len(n_keulegan)), n_keulegan])
    ng_df=ng_df.transpose()
    ng_df.columns=n_array_names
    ng_df.index=ng_df_index
    return (ng_df)

def compute_nt (Q_n, d_n_obs, nt_obs, nt_nd_range_top, nt_nd_range_bottom, CI):
    '''
    Parameters
    ----------
    Q_n: set of channel flow observations
    d_n_obs: set of channel depth observations
    nt_obs: set of total roughness (Manning's n) calculations
    nt_nd_range_top, nt_nd_range_bottom: range of non-dimensional n values 
    (nt/nt_bar) to be included
    CI: confidence interval 
    
    Returns
    -------
    a_n: power regression "a" value for relevant range
    b_n: power regression "b" value, recomputed for relevant range
    a_Df_n: power regression "a" value for depth, recomputed for relevant range
    b_Df_n: power regression "b" value for depth, recomputed for relevant range
    a_Uf_n: power regression "a" value for velocity, recomputed for relevant range
    b_Uf_n: power regression "b" value for velocity, recomputed for relevant range
    rsq: r-squared for non-dimensional nt versus d
    nt_obs_nd
    ind_n
    '''
    # Compute non-dimensional nt, weighted by x
    nt_obs_bar=sum(nt_obs*d_n_obs)/(sum(d_n_obs)) # depth-weighted average of nt
    nt_obs_nd=nt_obs/nt_obs_bar # non-dimensional total n
    
    # Select range of nt_nd
    ind_n=(nt_obs_nd>=nt_nd_range_top) & (nt_obs_nd<=nt_nd_range_bottom)
    
    # Recompute range of d and u for nt calculation and the d and u points and line
    d_1,d_0,dn_r, d_p, d_err=scipy.stats.linregress(np.log10(Q_n[ind_n]),
                                               np.log10(d_n_obs[ind_n]))
    a_Df_n=10**d_0
    b_Df_n=d_1
    Df_Q_n=a_Df_n*((Q_n)**b_Df_n)
    
    u_1, u_0, u_r, u_p, u_err=scipy.stats.linregress(np.log10(Q_n[ind_n]),
                                                 np.log10(u_n_obs[ind_n]))
    a_Uf_n=10**u_0
    b_Uf_n=u_1
    Uf_Q_n=a_Uf_n*((Q_n)**b_Uf_n)
    # Plot
    fig, (ax1, ax2)=plt.subplots(2,1,figsize=(6,8))
    ax1.plot(Q_n, d_n_obs,'ko',label='Observed, not used',
             markersize=8,markeredgecolor='black', markeredgewidth=1)
    ax1.plot(Q_n[ind_n], d_n_obs[ind_n],'ro',label='Observed, used',
             markersize=8,markeredgecolor='black', markeredgewidth=1)
    ax1.plot(np.sort(Q_n), Df_Q_n[np.argsort(Q_n)],'b-',label='Fitted Equation',
             markersize=8,markeredgecolor='black', linewidth=3)
    ax1.set_xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
    ax1.set_ylabel('Mean Stream Depth (m)',fontsize=12)
    ax1.legend(loc=4)
    ax1.set_title('Mean Stream Depth vs. Discharge',fontsize=12)
    ax1.tick_params(labelsize=12)
    ax1.grid(which='both')
    ax2.plot(Q_n, u_n_obs,'ko',label='Observed, not used',
             markersize=8,markeredgecolor='black', markeredgewidth=1)
    ax2.plot(Q_n[ind_n], u_n_obs[ind_n],'ro',label='Observed, used',
             markersize=8,markeredgecolor='black', markeredgewidth=1)
    ax2.plot(np.sort(Q_n), Uf_Q_n[np.argsort(Q_n)],'b-',label='Fitted Equation',
             markersize=8,markeredgecolor='black', linewidth=3)
    #plt.plot(Q_suite_inst[51:], u_inst[51:],'y^',label='Post-1998 Measurement')
    ax2.legend(loc='best')
    ax2.set_xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
    ax2.set_ylabel('Mean Stream Velocity (m/s)',fontsize=12)
    ax2.tick_params(labelsize=12)
    ax2.grid(which='both')
    ax2.set_title('Mean Stream Velocity vs. Discharge',fontsize=12)
    
    # Now do regression for the nt versus d
    p, y_err = lin_fit(np.log10(d_n_obs[ind_n]), np.log10(nt_obs_nd[ind_n]))
    a_n=10**p.c[1]
    b_n=p.c[0]
    d_1,d_0,d_r, d_p, d_err=scipy.stats.linregress(np.log10(d_n_obs[ind_n]),
                       np.log10(nt_obs_nd[ind_n]))
    rsq=d_r**2
    # Calculate confidence intervals
    p_x, confs = conf_calc(np.log10(d_n_obs[ind_n]), y_err, CI)
    p_y, lower, upper = ylines_calc(p, p_x, confs)
    # Plot these lines
    plt.figure()
    plt.plot(d_n_obs,nt_obs_nd,'ko',markersize=7, markeredgecolor='black', 
             label='Observations, not used')
    plt.plot(d_n_obs[ind_n],nt_obs_nd[ind_n],'ro',markersize=7, markeredgecolor='black', 
             label='Observations, used')
    plt.plot(np.sort(d_n_obs),a_n*np.sort(d_n_obs)**b_n,'b-',
         label='Regression line of Observations')
    plt.plot(10**p_x,10**lower,'b--',label='Regression line \n95% Confidence Limits')
    plt.plot(10**p_x,10**upper,'b--')
    plt.xlabel ('H (m)',fontsize=12)
    plt.ylabel('n$_{t*}$',fontsize=12)
    plt.title('n$_{t*}$ versus H with Power Regression Results',fontsize=12)
    plt.legend(loc='best')
    
    return (a_n, b_n, a_Df_n, b_Df_n, a_Uf_n, b_Uf_n, rsq, nt_obs_nd, ind_n)


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

def run_wc2003_all_obs (BLopng_inst, BL_sample_dis, tau_BL_obs, BL_sample_Fall, Wf):
    sg=2.65
    rho_w=1000
    Wstar_obs=pd.DataFrame(index=tau_BL_obs, columns=BLopng_inst)
    dis_m2s_obs=pd.DataFrame(index=tau_BL_obs, columns=BLopng_inst)
   # Convert bedload samples units from Mg/d to m3/s (1000 is for unit conversion since using rho+w=1000 kg/m3)    
    dis_m3s_BL_obs=BL_sample_dis*1000/(24*3600*sg*rho_w)
    dis_m2s_tot_obs=dis_m3s_BL_obs/Wf
    
    for i in np.arange (0,len(dis_m2s_obs.index)):
        dis_m2s_obs.loc[dis_m2s_obs.index[i], :]=dis_m2s_tot_obs[i]*BL_sample_Fall[i,:]/100
        for j in np.arange (0,(len(Wstar_obs.columns))):
            if BL_sample_Fall[i,j]==0:
                Wstar_obs.loc[Wstar_obs.index[i],Wstar_obs.columns[j]]=0
            else: Wstar_obs.loc[Wstar_obs.index[i],Wstar_obs.columns[j]]=(sg-1)*g*dis_m2s_obs.loc[dis_m2s_obs.index[i],dis_m2s_obs.columns[j]]/((Chdia_inst_WC_fractions[j]/100)*(((Wstar_obs.index[i]/rho_w)**0.5)**3))    
    Wstar_tot_obs=Wstar_obs.sum(axis=1)
    return Wstar_obs, Wstar_tot_obs, dis_m2s_tot_obs, dis_m3s_BL_obs

def run_wc2003_all_model (Ch_Fs, Ch_Dmean_m, Chopng_inst, tau_BL_obs):
    sg=2.65
    rho_w=1000
    # Constants
    A_WC=14
    chi_WC=0.894
    exp_WC=0.5
    phi_prime_WC=1.35
    tau_star_rsm_WC=0.021+0.015*np.exp(-20*Ch_Fs)
    tau_rsm_WC=tau_star_rsm_WC*(sg-1)*rho_w*g*Ch_Dmean_m

    # 1 value per sieve size (26 values)
    b=0.67/(1+np.exp(1.5-(Chopng_inst/(1000*Ch_Dmean_m))))
    tau_ri=tau_rsm_WC*(Chopng_inst/(1000*Ch_Dmean_m))**b
    
    # Model
    phi_line=np.arange(0.1,20,0.1)
    Wstar_line=np.empty(len(phi_line))
    for i in np.arange (0,(len(phi_line))):
        if phi_line[i] < phi_prime_WC:
            Wstar_line[i]=0.002*(phi_line[i])**7.5
        elif chi_WC/((phi_line[i])**exp_WC)>1: # Checka that term in paraentheses is not negative
            Wstar_line[i]=0
        else:
            Wstar_line[i]=A_WC*((1-(chi_WC/((phi_line[i])**exp_WC)))**(4.5))
    
    # Modeled Samples
    phi_WC=pd.DataFrame(index=tau_BL_obs, columns=Chopng_inst)
    Wstar_WC=pd.DataFrame(index=tau_BL_obs, columns=Chopng_inst)
    dis_m2s_WC=pd.DataFrame(index=tau_BL_obs, columns=Chopng_inst)    
    for i in np.arange (0,len(Wstar_WC.index)): # for each row (tau)
        phi_WC.loc[phi_WC.index[i], :]=Wstar_WC.index[i]/tau_ri
        for j in np.arange (0,len(Wstar_WC.columns)):
            if phi_WC.loc[phi_WC.index[i], phi_WC.columns[j]] < phi_prime_WC:
                Wstar_WC.loc[Wstar_WC.index[i], Wstar_WC.columns[j]]=0.002*(phi_WC.loc[phi_WC.index[i], phi_WC.columns[j]])**7.5
                dis_m2s_WC.loc[dis_m2s_WC.index[i], dis_m2s_WC.columns[j]]=(((dis_m2s_WC.index[i]/rho_w)**0.5)**3)*Wstar_WC.loc[dis_m2s_WC.index[i], Wstar_WC.columns[j]]*(Chdia_inst_WC_fractions[j]/100)/((sg-1)*g)
            elif chi_WC/((phi_WC.loc[phi_WC.index[i], phi_WC.columns[j]])**exp_WC)>1: # Checka that term in paraentheses is not negative
               Wstar_WC.loc[Wstar_WC.index[i], Wstar_WC.columns[j]]=0
               dis_m2s_WC.loc[dis_m2s_WC.index[i], dis_m2s_WC.columns[j]]=0
            else:
                Wstar_WC.loc[Wstar_WC.index[i], Wstar_WC.columns[j]]=A_WC*((1-(chi_WC/((phi_WC.loc[phi_WC.index[i], phi_WC.columns[j]])**exp_WC)))**(4.5))    
                dis_m2s_WC.loc[dis_m2s_WC.index[i], dis_m2s_WC.columns[j]]=(((Wstar_WC.index[i]/rho_w)**0.5)**3)*Wstar_WC.loc[Wstar_WC.index[i], Wstar_WC.columns[j]]*(Chdia_inst_WC_fractions[j]/100)/((sg-1)*g)
    
    Wstar_tot_WC=Wstar_WC.sum(axis=1)
    dis_m2s_tot_WC= dis_m2s_WC.sum(axis=1)

    return (phi_line, Wstar_line, phi_WC, Wstar_tot_WC, dis_m2s_tot_WC)

def run_wc2003_2F_obs (BL_sample_dis, BL_sample_Fg, BL_sample_Fs, tau_BL_obs, Wf):
    sg=2.65
    rho_w=1000
    # Convert bedload samples units from Mg/d to m3/s (1000 is for unit conversion since using rho+w=1000 kg/m3)
    dis_m3s_BL_obs=BL_sample_dis*1000/(24*3600*sg*rho_w)
    
    # Convert bedload samples units from m3/s to m2/s
    dis_m2s_BL_obs=dis_m3s_BL_obs/Wf
    
    # Calculate Wi* for comparison to Wilcock equations
    dis_m2s_gravel_obs=BL_sample_Fg*dis_m2s_BL_obs
    dis_m2s_sand_obs=BL_sample_Fs*dis_m2s_BL_obs
    
    u_star_obs=(tau_BL_obs/rho_w)**0.5
    Wstar_gravel_obs=(sg-1)*g*dis_m2s_gravel_obs/(Ch_Fg*u_star_obs**3)
    Wstar_sand_obs=(sg-1)*g*dis_m2s_sand_obs/(Ch_Fs*u_star_obs**3)
    
    return (Wstar_gravel_obs, Wstar_sand_obs, dis_m2s_gravel_obs, 
            dis_m2s_sand_obs, dis_m2s_BL_obs, dis_m3s_BL_obs)

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
    
#%% IMPORT AND FORMAT DATA
os.chdir(datadir)

LM_all_DF= pd.read_excel('Mills_AllData_Merged.xlsx',sheetname='12044900_daily',
                         header=3, skiprows=[0,1,2]) # import data
# Header: Row 3 in Python, Row 7 in Excel- Describes variables
# Skip rows 0-1 in Python, 1-6 in Excel (rows 3-6 are merged)
LM_all=pd.DataFrame.as_matrix(LM_all_DF) # convert to Numpy array
# Extract Dates- get year, month, day into datetime objects:
n=len(LM_all[:,0]) # n is number of entries in the record
datetime_1=np.full(n,'',dtype=object) # Preallocate date matrix
for x in range(0,n): # Cycle through all days of the year
    datetime_1[x]=datetime.date(int(LM_all[x,2]),int(LM_all[x,3]), int(LM_all[x,4]))

# Extract Data (description of each variable included below) and remove nan 
# values. Also extract the date/time for each entry
# Varaibles with "_all" means that array includes "nan" values where no data 
# Variables without "_all" means that array only includes where there is data 
# Each variable has a datetime matrix corresponding to date/time that sample collected
# ind_ is variable of indices when sample collected

# Streamflow
Q_day_all=np.array((LM_all[:,7]), dtype='float64') # mean daily streamflow, m^3/s - for all values
ind_Q_day=~np.isnan(Q_day_all) # Find indices where have a value
Q_day=Q_day_all[ind_Q_day] # Only include indices where have a value
datetime_Q_day=datetime_1[ind_Q_day]

Q_inst_all=np.array((LM_all[:,8]), dtype='float64') # instantaneous streamflow, m^3/s 
ind_Q_inst=~np.isnan(Q_inst_all) # Find indices where have a value
Q_inst=Q_inst_all[ind_Q_inst] # Only include indices where have a value
datetime_Q_inst=datetime_1[ind_Q_inst]

temp_inst_all=np.array((LM_all[:,9]), dtype='float64') # instantaneous temperature, deg C
ind_temp_inst=~np.isnan(temp_inst_all) # Find indices where have a value
temp_inst=temp_inst_all[ind_temp_inst] # Only include indices where have a value
datetime_temp_inst=datetime_1[ind_temp_inst]

# Hydraulic Geometry
w_inst_all=np.array(LM_all[:,10], dtype='float64') # stream width (m)
ind_w_inst=~np.isnan(w_inst_all) # Find indices where have a value
w_inst=w_inst_all[ind_w_inst] # Only include indices where have a value
datetime_w_inst=datetime_1[ind_w_inst]

d_inst_all=np.array(LM_all[:,11], dtype='float64') # mean stream depth (m)
ind_d_inst=~np.isnan(d_inst_all) # Find indices where have a value
d_inst=d_inst_all[ind_d_inst] # Only include indices where have a value
datetime_d_inst=datetime_1[ind_d_inst]

u_inst_all=np.array(LM_all[:,12], dtype='float64') # mean velocity (m/s)
ind_u_inst=~np.isnan(u_inst_all) # Find indices where have a value
u_inst=u_inst_all[ind_u_inst] # Only include indices where have a value
datetime_u_inst=datetime_1[ind_u_inst]

A_inst_all=np.array(LM_all[:,13], dtype='float64') # cross-section area (m2)
ind_A_inst=~np.isnan(A_inst_all) # Find indices where have a value
A_inst=A_inst_all[ind_A_inst] # Only include indices where have a value
datetime_A_inst=datetime_1[ind_A_inst]

# Suspended Sediment/Turbidity
SSC_day_all=np.array(LM_all[:,14], dtype='float64') # mean daily SSC,  mg/L
ind_SSC_day=~np.isnan(SSC_day_all) # Find indices where have a value
SSC_day=SSC_day_all[ind_SSC_day] # Only include indices where have a value
datetime_SSC_day=datetime_1[ind_SSC_day]

SSL_day_all=np.array(LM_all[:,15], dtype='float64') # mean daily suspended sediment load, tons/day
ind_SSL_day=~np.isnan(SSL_day_all) # Find indices where have a value
SSL_day=SSL_day_all[ind_SSL_day] # Only include indices where have a value
datetime_SSL_day=datetime_1[ind_SSL_day]

SSC_inst_all=np.array(LM_all[:,16], dtype='float64') # instantaneous SSC,  mg/L
ind_SSC_inst=~np.isnan(SSC_inst_all) # Find indices where have a value
SSC_inst=SSC_inst_all[ind_SSC_inst] # Only include indices where have a value
datetime_SSC_inst=datetime_1[ind_SSC_inst]

SSL_inst_all=np.array(LM_all[:,17], dtype='float64') # instantaneous suspended sediment load, tons/day
ind_SSL_inst=~np.isnan(SSL_inst_all) # Find indices where have a value
SSL_inst=SSL_inst_all[ind_SSL_inst] # Only include indices where have a value
datetime_SSL_inst=datetime_1[ind_SSL_inst]

# NOTE: SUSPENDED SEDIMENT SAMPLE GRADATIONS MUST HAVE A VALUE IN COLUMN 23
# FOR RELEVANT SAMPLES (% LESS THAN 0.063) FOR THIS CODE TO WORK!
SSdia_inst_all=np.array(LM_all[:,18:24], dtype='float64') # Suspended sediment- percent finer than diameter in SSopng_inst
ind_SSdia_inst=~np.isnan(SSdia_inst_all[:,-1]) # Find indices where have a value
SSdia_inst=SSdia_inst_all[ind_SSdia_inst,:] # Only include indices where have a value
datetime_SSdia_inst=datetime_1[ind_SSdia_inst]
q_SSdia_inst=Q_inst_all[ind_SSdia_inst]

SSopng_inst=np.array([2.00, 1.00, 0.500, 0.25, 0.125, 0.063],dtype='float64') # Diameter of suspended sediment sieve openings

T_day_all=np.array(LM_all[:,24], dtype='float64') # Daily meDIAN turbidity, FNU
ind_T_day=~np.isnan(T_day_all) # Find indices where have a value
T_day=T_day_all[ind_T_day] # Only include indices where have a value
datetime_T_day=datetime_1[ind_T_day]

T_inst_all=np.array(LM_all[:,25], dtype='float64') # Instantaneous turbidity, NTU
ind_T_inst=~np.isnan(T_inst_all) # Find indices where have a value
T_inst=T_inst_all[ind_T_inst] # Only include indices where have a value
datetime_T_inst=datetime_1[ind_T_inst]

# Bedload
BLnum_inst_all=np.array(LM_all[:,26], dtype='float64') # Bedload discharge, Mg/day
ind_BLnum_inst=~np.isnan(BLnum_inst_all) # Find indices where have a value
BLnum_inst=BLnum_inst_all[ind_BLnum_inst] # Only include indices where have a value
datetime_BLnum_inst=datetime_1[ind_BLnum_inst]

BLdis_inst_all=np.array(LM_all[:,27], dtype='float64') # Bedload discharge, Mg/day
ind_BLdis_inst=~np.isnan(BLdis_inst_all) # Find indices where have a value
BLdis_inst=BLdis_inst_all[ind_BLdis_inst] # Only include indices where have a value
datetime_BLdis_inst=datetime_1[ind_BLdis_inst]

BLwdis_inst=Q_inst_all[ind_BLdis_inst] # Streamflow discharge when bedload sample collected

# NOTE: BEDLOAD GRADATIONS MUST HAVE A VALUE IN BEDLOAD SAMPLE NUMBER COLUMN 
# FOR RELEVANT SAMPLES FOR THIS CODE TO WORK!
BLdia_inst_all=np.array(LM_all[:,28:54], dtype='float64') # Bedload- percent finer than diameter in BLopng_inst
BLdia_inst=BLdia_inst_all[ind_BLnum_inst,:] # Only include indices where have a value
datetime_BLdia_inst=datetime_1[ind_BLnum_inst]

BLopng_inst=np.array([362, 256, 181, 128, 90.5, 64, 45.2, 32.0, 22.6, 16.0, 
                        11.3, 8.00, 5.66, 4.00, 2.83, 2.00, 1.41, 1.00, 0.708, 
                        0.5, 0.354, 0.250, 0.177, 0.125, 0.088, 0.062],
                        dtype='float64') # Diameter of bedload sieve openings, mm

# Channel Surface/Gravel bar surface/ Gravel bar subsurface- 
# NOTE: Rows 0-4 are wetted channel surface, 
# rows 5-6 are gravel bar surface, 
# row 7 is gravel bar subsurface
Chnum_inst_all=np.array(LM_all[:,57], dtype='float64') # Wetted Channel Surface sample numbers
ind_Chnum_inst=~np.isnan(Chnum_inst_all) # Find indices where have a value
Chnum_inst=Chnum_inst_all[ind_Chnum_inst] # Only include indices where have a value
datetime_Chnum_inst=datetime_1[ind_Chnum_inst]

Chdia_inst_all=np.array(LM_all[:,58:84], dtype='float64') # Wetted Channel Surface- percent finer than diameter in BLopng_inst
ind_Chdia_inst=~np.isnan(Chnum_inst_all) # Find indices where have a value
Chdia_inst=Chdia_inst_all[ind_Chdia_inst,:] # Only include indices where have a value
datetime_Chdia_inst=datetime_1[ind_Chdia_inst]

Chopng_inst=np.array([362, 256, 181, 128, 90.5, 64, 45.2, 32.0, 22.6, 16.0, 
                        11.3, 8.00, 5.66, 4.00, 2.83, 2.00, 1.41, 1.00, 0.708, 
                        0.5, 0.354, 0.250, 0.177, 0.125, 0.088, 0.062],
                        dtype='float64') # Diameter of wetted channel sieve openings, mm
            
# Delete datasets that are not used          
del(Q_day_all, ind_Q_day, ind_Q_inst, temp_inst_all, ind_temp_inst, d_inst_all,
    u_inst_all, ind_u_inst, A_inst_all, ind_A_inst, SSC_day_all, ind_SSC_day, 
    SSL_day_all, ind_SSL_day, ind_SSL_inst, SSdia_inst_all, 
    T_day_all, ind_T_day, T_inst_all, ind_T_inst, ind_BLnum_inst, 
    BLdis_inst_all, BLdia_inst_all, Chnum_inst_all, ind_Chnum_inst, 
    Chdia_inst_all, ind_Chdia_inst)
#%%
# import modeled streamflow
os.chdir(homedir)
q_mod=pickle.load(open('streamflow.py', 'rb'))
q_mod=q_mod[144][datetime.date(1927,1,1):datetime.date(2010,12,31)]

#%% Streamflow Rating Curve, Sediment, and Channel Bed Properties
# Compute streamflow rating curve
# Extract flow data for time period of interest 
FDC_1_st=np.where(datetime_Q_day==FDC_1_st_date)[0]  # CY start for FDC pt 1
FDC_1_ed=np.where(datetime_Q_day==FDC_1_ed_date)[0]  # CY end for FDC pt 1
FDC_2_st=np.where(datetime_Q_day==FDC_2_st_date)[0]  # CY start for FDC pt 2
FDC_2_ed=np.where(datetime_Q_day==FDC_2_ed_date)[0]  # CY end for FDC pt 2              
Q_curve=Q_day[FDC_1_st[0]:FDC_1_ed[0]]
Q_curve=np.append(Q_curve, Q_day[FDC_2_st[0]:FDC_2_ed[0]])
    
# Check if there are null values in dataset- array should be empty! If array
# is not empty, disclude the values indicated
check=np.where(np.isnan(Q_curve))
print('Check for null values in array- returned array should be empty:',check)
############
#Q_RC=np.sort(Q_curve) # sorts from lowest to highest value
Q_RC=np.sort(q_mod)
###########

n_curve=len(Q_RC)
cum_pcntl=np.array(range(1,n_curve+1))/(n_curve+1) # Compute cumulative percentile using Weibull Plotting Position
dp=cum_pcntl[1]-cum_pcntl[0] # Compute change in percent for each step

# Compute streamflow magnitudes: 5th, 50th, and 95th percentile
Q_RC_5_m3s=Q_RC[np.where(cum_pcntl==find_nearest(cum_pcntl, 0.05))] # 5th percentile
Q_RC_5_mmday=Q_RC_5_m3s/A_LM*3600*24*1000
print('5th percentile flow (cms):',Q_RC_5_m3s)
Q_RC_50_m3s=Q_RC[np.where(cum_pcntl==find_nearest(cum_pcntl, 0.50))] # 50th percentile
Q_RC_50_mmday=Q_RC_50_m3s/A_LM*3600*24*1000
print('50th percentile flow (cms):',Q_RC_50_m3s)
Q_RC_95_m3s=Q_RC[np.where(cum_pcntl==find_nearest(cum_pcntl, 0.95))] # 95th percentile
Q_RC_95_mmday=Q_RC_95_m3s/A_LM*3600*24*1000
print('95th percentile flow (cms):',Q_RC_95_m3s)

# Compute characteristic grain sizes of bedload and channel surface samples
# Compute d50, d65, d90 for each bedload sample using interpolation (of bedload substrate)
BLtotalsamples=len(BLnum_inst) # number of individual bedload samples
BL_d50=np.empty([BLtotalsamples]) # preallocate arrays
BL_d65=np.empty([BLtotalsamples])
BL_d84=np.empty([BLtotalsamples])
BL_d90=np.empty([BLtotalsamples])
BL_Fs=np.empty([BLtotalsamples])
BL_Fg=np.empty([BLtotalsamples])

for i in range (0, BLtotalsamples):
    dist_i=BLdia_inst[i,:] # Extract distribution of 1 sample
    if np.isnan(dist_i).all()==True: # if there are no entries in the distribution, d50, etc. is blank
        BL_d50[i]=None
        BL_d65[i]=None
        BL_d90[i]=None
        BL_Fs[i]=None
        BL_Fg[i]=None
    else: # Look for upper and lower bound for interpolation
        # d50
        t_50=np.where(dist_i>50)[0] 
        t_50=t_50[-1]
        b_50=np.where(dist_i<50)[0]
        b_50=b_50[0]
        compute_d50=interp1d([dist_i[t_50], dist_i[b_50]],
                             [BLopng_inst[t_50], BLopng_inst[b_50]])
        BL_d50[i]=compute_d50(50)
        # d65
        t_65=np.where(dist_i>65)[0]
        t_65=t_65[-1]
        b_65=np.where(dist_i<65)[0]     
        b_65=b_65[0]
        compute_d65=interp1d([dist_i[t_65], dist_i[b_65]],
                             [BLopng_inst[t_65], BLopng_inst[b_65]])
        BL_d65[i]=compute_d65(65)
        # d84
        t_84=np.where(dist_i>84)[0]
        t_84=t_84[-1]
        b_84=np.where(dist_i<84)[0]     
        b_84=b_84[0]
        compute_d84=interp1d([dist_i[t_84], dist_i[b_84]],
                             [BLopng_inst[t_84], BLopng_inst[b_84]])
        BL_d84[i]=compute_d84(84)
        # d90
        t_90=np.where(dist_i>90)[0]
        t_90=t_90[-1]
        b_90=np.where(dist_i<90)[0]     
        b_90=b_90[0]
        compute_d90=interp1d([dist_i[t_90], dist_i[b_90]],
                             [BLopng_inst[t_90], BLopng_inst[b_90]])
        BL_d90[i]=compute_d90(90)
        # Fs and Fg
        # Assume that all bedload has the same bulk density
        BL_Fs[i]=dist_i[np.where(BLopng_inst==2.00)[0][0]]/100 # average fraction of sand
        BL_Fg[i]=1-BL_Fs[i] # average fraction of gravel

## Plot distribution curves
BLplots=plt.figure(figsize=(4,3))
j=1    
# 
for i in range(0,len(BLdia_inst)):
    if BLnum_inst[i]==j:
        row=BLdia_inst[i,:]
        start=np.where(row==100)[-1][-1]
        y=row[start::]
        x=BLopng_inst[start::]
        x=x[~np.isnan(y)]
        y=y[~np.isnan(y)]
        plt.semilogx(x,y,'b-')
        plt.semilogx([0.0625, 0.0625], [0, 100],'k--')
        plt.semilogx([2, 2], [0, 100], 'k--', markersize=5)
        plt.semilogx([64, 64], [0, 100],'k--', markersize=5)         
    else:
        if BLnum_inst[i]==21.0: # make exceptions for blank size distributions
            j=22
        elif BLnum_inst[i]==33.0:
            j=34
        else: 
            j=j+1
            x=BLopng_inst
            y=BLdia_inst[i,:]
            x=x[~np.isnan(y)]
            y=y[~np.isnan(y)]
            plt.semilogx(x,y,'b-', linewidth=1)
            plt.semilogx([0.0625, 0.0625], [0, 100],'k--', markersize=2)
            plt.semilogx([2, 2], [0, 100],'k--',markersize=2)
            plt.semilogx([64, 64], [0, 100],'k--',markersize=2)        
plt.xlabel ('Particle Diameter (mm)',fontsize=12)
plt.ylabel('Percent Finer (%)',fontsize=12)
#plt.title('Bedload Discharge Sample Distributions',fontsize=12)
plt.xlim(10**-2,10**3)
plt.ylim(0,100)
plt.tick_params(labelsize=12)

# Bed Surface Calculations
# Compute median grain size of bed surface samples
Chtotalsamples=len(Chdia_inst) # Count the number of samples
Ch_dsm=np.empty([Chtotalsamples]) # preallocate array for mean channel surface size [mm]
Ch_d16=np.empty([Chtotalsamples]) # preallocate array for d16 of channel surface size [mm]
Ch_d50=np.empty([Chtotalsamples]) # preallocate array for median channel surface size [mm]
Ch_d65=np.empty([Chtotalsamples]) # preallocate array for d65 of channel surface size [mm]
Ch_d84=np.empty([Chtotalsamples]) # preallocate array for d84 of channel surface size [mm]
Ch_d90=np.empty([Chtotalsamples]) # preallocate array for d90 of channel surface size [mm]
Ch_Dsand=np.empty([Chtotalsamples]) # preallocate array for median sand size (less than 2mm diameter) [mm]
Ch_Dgravel=np.empty([Chtotalsamples]) # preallocate array for median gravel size (greater than 2mm diameter) [mm]
ind_sand=np.where(Chopng_inst==2.00)[0][0] # Find the entry number for the sand size

for i in range (0, Chtotalsamples):
    dist_i=Chdia_inst[i,:] # Extract distribution
    # Compute mean grain size (dsm) of channel bed
    start=np.where(dist_i==100)[-1][-1]
    temp=((dist_i[start:-1]-dist_i[start+1::])/100)*((Chopng_inst[start:-1]+Chopng_inst[start+1::])/2)
    Ch_dsm[i]=np.nansum(temp,axis=0)
   
    # Compute d16 of channel bed
    t_16=np.where(dist_i>16)[0] # Look for upper bound for interpolation
    t_16=t_16[-1]
    b_16=np.where(dist_i<16)[0] # Look for lower bound for interpolation
    b_16=b_16[0]
    compute_d16=interp1d([dist_i[t_16], dist_i[b_16]], [Chopng_inst[t_16], Chopng_inst[b_16]]) # interpolation function
    Ch_d16[i]=compute_d16(16) # interpolate

    # Compute median grain size (d50) of channel bed
    t_50=np.where(dist_i>50)[0] # Look for upper bound for interpolation
    t_50=t_50[-1]
    b_50=np.where(dist_i<50)[0] # Look for lower bound for interpolation
    b_50=b_50[0]
    compute_d50=interp1d([dist_i[t_50], dist_i[b_50]], [Chopng_inst[t_50], Chopng_inst[b_50]]) # interpolation function
    Ch_d50[i]=compute_d50(50) # interpolate
    
    # Compute d65 of channel bed
    t_65=np.where(dist_i>65)[0]
    t_65=t_65[-1]
    b_65=np.where(dist_i<65)[0]     
    b_65=b_65[0]        
    compute_d65=interp1d([dist_i[t_65], dist_i[b_65]], [Chopng_inst[t_65], Chopng_inst[b_65]])
    Ch_d65[i]=compute_d65(65)
   
    # Compute d84 of channel bed
    t_84=np.where(dist_i>84)[0]
    t_84=t_84[-1]
    b_84=np.where(dist_i<84)[0]     
    b_84=b_84[0]        
    compute_d84=interp1d([dist_i[t_84], dist_i[b_84]], [Chopng_inst[t_84], Chopng_inst[b_84]])
    Ch_d84[i]=compute_d84(84)
   
    # Compute d90 of channel bed
    t_90=np.where(dist_i>90)[0]
    t_90=t_90[-1]
    b_90=np.where(dist_i<90)[0]     
    b_90=b_90[0]        
    compute_d90=interp1d([dist_i[t_90], dist_i[b_90]], [Chopng_inst[t_90], Chopng_inst[b_90]])
    Ch_d90[i]=compute_d90(90)
    
    # Compute median grain size of gravel
    gravel_mid=dist_i[ind_sand]+(100-dist_i[ind_sand])/2
    t_Dgravel=np.where(dist_i>gravel_mid)[0]
    t_Dgravel=t_Dgravel[-1]
    b_Dgravel=np.where(dist_i<gravel_mid)[0]
    b_Dgravel=b_Dgravel[0]
    compute_Dgravel=interp1d([dist_i[t_Dgravel], dist_i[b_Dgravel]],
                         [Chopng_inst[t_Dgravel], Chopng_inst[b_Dgravel]])
    Ch_Dgravel[i]=compute_Dgravel(gravel_mid)
    # Compute median grain size of sand
    if np.isnan(dist_i[ind_sand+1]):
        Ch_Dsand[i]=2
    else:
        sand_mid=(dist_i[ind_sand])/2
        t_Dsand=np.where(dist_i>sand_mid)[0]
        t_Dsand=t_Dsand[-1]
        b_Dsand=np.where(dist_i<sand_mid)[0]
        b_Dsand=b_Dsand[0]
        compute_Dsand=interp1d([dist_i[t_Dsand], dist_i[b_Dsand]],
                             [Chopng_inst[t_Dsand], Chopng_inst[b_Dsand]])
        Ch_Dsand[i]=compute_Dsand(sand_mid)

# Plot surface distribution
plt.semilogx(BLopng_inst,BLdia_inst[i,:],'b-', linewidth=1, label='Bedload Substrate')
plt.semilogx(Chopng_inst,Chdia_inst[0,:],'r*-',linewidth=2, label='Wetted Channel Surface')
plt.semilogx(Chopng_inst,Chdia_inst[1,:],'r*-',linewidth=2)
plt.semilogx(Chopng_inst,Chdia_inst[2,:],'r*-',linewidth=2)
plt.semilogx(Chopng_inst,Chdia_inst[3,:],'r*-',linewidth=2)
plt.semilogx(Chopng_inst,Chdia_inst[4,:],'r*-',linewidth=2)
plt.semilogx(Chopng_inst,Chdia_inst[5,:],'gs-',linewidth=2, label='Upstream Gravel Bar')
plt.semilogx(Chopng_inst,Chdia_inst[6,:],'co-',linewidth=2, label='Downstream Gravel Bar')

plt.semilogx([0.0625, 0.0625], [0, 100],'k--')
plt.semilogx([2, 2], [0, 100],'k--')
plt.semilogx([64, 64], [0, 100],'k--')  
#plt.legend(loc='best')
plt.xlim(10**-2,10**3)
plt.ylim(0,100)
plt.xlabel ('Particle Diameter (mm)',fontsize=12)
plt.ylabel('Percent Finer (%)',fontsize=12)
#plt.title('Bed Material Particle Distributions',fontsize=12)
   #%%      
# BEDLOAD SAMPLES: Compute mean parameter for each bedload sample 
# (since samples were collected acoss a cross section)
BL_sample_n=int(BLnum_inst[-1])
BL_sample_date=np.empty([BL_sample_n], dtype=object)  # sample date
BL_sample_month=np.empty([BL_sample_n], dtype=object)  # sample month             
BL_sample_wdis=np.empty([BL_sample_n])                # streamflow, m3/s
BL_sample_dis=np.empty([BL_sample_n])                 # bedload discharge, Mg/d
BL_sample_d50_mm=np.empty([BL_sample_n])              # d50 diameter, mm
BL_sample_d65_mm=np.empty([BL_sample_n])              # d65 diameter, mm
BL_sample_d84_mm=np.empty([BL_sample_n])              # d65 diameter, mm
BL_sample_d90_mm=np.empty([BL_sample_n])              # d90 diameter, mm
BL_channelw_indv=w_inst_all[ind_BLdis_inst]       # Channel width of individual bedload samples
BL_sample_channelw=np.empty([BL_sample_n])            # channel width, m (samples 21-45 only)
BL_sample_Fs=np.empty([BL_sample_n])                  # fraction of sand
BL_sample_Fg=np.empty([BL_sample_n])                  # fraction of gravel
BL_sample_Favg=np.empty([BL_sample_n,len(BLopng_inst)]) # average perecnt in each sample
BL_sample_Fall=np.empty([BL_sample_n,len(BLopng_inst)]) # fraction of all grain sizes in BLopng_inst

for i in range (1, BL_sample_n+1):
    temp_ind=np.where(BLnum_inst==i); # find index of sample
    BL_sample_date[i-1]=datetime_BLdis_inst[(temp_ind[0][0])]
    BL_sample_month[i-1]=BL_sample_date[i-1].month
    # Use mean of streamflow discharge, bedload discharge and channel width
    BL_sample_wdis[i-1]=np.nanmean(BLwdis_inst[(temp_ind[0])])
    BL_sample_dis[i-1]=np.nanmean(BLdis_inst[(temp_ind[0])])
    BL_sample_channelw[i-1]=np.nanmean(BL_channelw_indv[(temp_ind[0])]) # causes error for samples that do not have widths (childers data)
    # Use weighted mean (by bedload discharge) for d50, d65, d90, fraction of sand and fraction of gravel  
    BL_sample_d50_mm[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_d50[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_d65_mm[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_d65[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_d84_mm[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_d84[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_d90_mm[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_d90[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_Fs[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_Fs[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_Fg[i-1]=np.nansum(BLdis_inst[(temp_ind[0])]*BL_Fg[(temp_ind[0])])/(np.nansum(BLdis_inst[(temp_ind[0])]))
    BL_sample_Favg[i-1,:]=BLdia_inst[(temp_ind[0])].mean(axis=0)
    BL_sample_Fall[i-1,:]=BL_sample_Favg[i-1,:]-np.concatenate((BL_sample_Favg[i-1,:][1::], [0])) 

# Delete samples that have missing streamflow (19) and size distribution (20 & 32)
BL_sample_n=BL_sample_n-3
BL_sample_date=np.delete(BL_sample_date,[19, 20, 32])
BL_sample_month=np.delete(BL_sample_month,[19, 20, 32])
BL_sample_wdis=np.delete(BL_sample_wdis,[19, 20, 32])
BL_sample_dis=np.delete(BL_sample_dis,[19, 20, 32])
BL_sample_d50_mm=np.delete(BL_sample_d50_mm,[19, 20, 32])
BL_sample_d65_mm=np.delete(BL_sample_d65_mm,[19, 20, 32])
BL_sample_d84_mm=np.delete(BL_sample_d84_mm,[19, 20, 32])
BL_sample_d90_mm=np.delete(BL_sample_d90_mm,[19, 20, 32])
BL_sample_channelw=np.delete(BL_sample_channelw,[19, 20, 32])
BL_sample_Fs=np.delete(BL_sample_Fs,[19, 20, 32])
BL_sample_Fg=np.delete(BL_sample_Fg,[19, 20, 32])
BL_sample_Favg=np.delete(BL_sample_Favg,[19, 20, 32], axis=0)
BL_sample_Fall=np.delete(BL_sample_Fall,[19, 20, 32], axis=0)

BL_sample_d50_m=BL_sample_d50_mm/1000
BL_sample_d65_m=BL_sample_d65_mm/1000
BL_sample_d84_m=BL_sample_d84_mm/1000
BL_sample_d90_m=BL_sample_d90_mm/1000

# CHANNEL BED SURFACE
Chdiam_all=Chdia_inst[0:7] # 0:7 to include everything except sub-surface sample
Chdia_inst_WC=np.vstack((Chdiam_all[0:5], Chdiam_all[6:7]))
Chdia_inst_WC_avg=Chdia_inst_WC.mean(axis=0)
Chdia_inst_WC_fractions=Chdia_inst_WC_avg-np.concatenate((Chdia_inst_WC_avg[1::], [0]))
Ch_d16_m=np.mean(np.concatenate((Ch_d16[0:5], Ch_d16[6:7])))/1000 # d65 grain size of bed surface in [m]
Ch_d50_m=np.mean(np.concatenate((Ch_d50[0:5], Ch_d50[6:7])))/1000 # median grain size of bed surface in [m]
Ch_d65_m=np.mean(np.concatenate((Ch_d65[0:5], Ch_d65[6:7])))/1000 # d65 grain size of bed surface in [m]
Ch_d84_m=np.mean(np.concatenate((Ch_d84[0:5], Ch_d84[6:7])))/1000 # d84 grain size of bed surface in [m]
Ch_d90_m=np.mean(np.concatenate((Ch_d90[0:5], Ch_d90[6:7])))/1000 # d90 grain size of bed surface in [m]
Ch_Dgravel_m=np.mean(np.concatenate((Ch_Dgravel[0:5], Ch_Dgravel[6:7])))/1000 # mean grain size of bed surface gravel portion in [m]
Ch_Dsand_m=np.mean(np.concatenate((Ch_Dsand[0:5], Ch_Dsand[6:7])))/1000 # mean grain size of bed surface sand portion in [m]
Ch_Fs=np.mean(Chdia_inst_WC[:,np.where(Chopng_inst==2.00)[0][0]])/100 # average fraction of sand
Ch_Fg=1-Ch_Fs # average fraction of gravel   
Ch_Dmean_rows=np.empty(len(Chdia_inst_WC))
for i in range (0,len(Chdia_inst_WC)):
    row=Chdia_inst_WC[i]
    start=np.where(row==100)[0][0]
    end=np.where(row==np.nanmin(row))[0][-1]
    weighted_size=np.empty(end-start)
    for j in range (start,end):
        avg_size=(Chopng_inst[j]+Chopng_inst[j+1])/2
        percent=(row[j]-row[j+1])/100
        weighted_size[j-start]=avg_size*percent
    Ch_Dmean_rows[i]=np.sum(weighted_size)
Ch_Dmean_m=np.nanmean(Ch_Dmean_rows)/1000
Ch_sigma=0.5*((Ch_d84_m/Ch_d50_m)+(Ch_d16_m/Ch_d50_m))

print('Ch_d16 (mm)=',Ch_d16_m*1000)
print('Ch_d50 (mm)=',Ch_d50_m*1000)
print('Ch_d65 (mm)=',Ch_d65_m*1000)
print('Ch_d84 (mm)=',Ch_d84_m*1000)
print('Ch_d90 (mm)=',Ch_d90_m*1000)
print('Ch_Dmean (mm)=',Ch_Dmean_m*1000)
print('Ch_Dgravel (mm)=',Ch_Dgravel_m*1000)
print('Ch_Dsand (mm)=',Ch_Dsand_m*1000)
print('Ch_Fs (mm)=',Ch_Fs)
print('Ch_Fg (mm)=',Ch_Fg)
print('Ch_sigma=',Ch_sigma)

#%% Hydraulic geometry
# Hydraulic geometry relationships are based on the logarithm base 10 of 
# streamflow and hydraulic parameters since they are expected to be related in 
# the form of simple power functions.
#
# Compute power regression parameters for width, depth, and velocity and then 
# apply them to streamflow samples along with bedload samples 
#   
w_suite_inst=w_inst_all[ind_d_inst]
Q_suite_inst=Q_inst_all[ind_d_inst]
Q_suite_inst_sort=np.sort(Q_suite_inst)
Q_suite_inst_sort_ind=np.argsort(Q_suite_inst)
w_suite_inst_sort= w_suite_inst[Q_suite_inst_sort_ind]
datetime_suite_inst=datetime_d_inst
datetime_suite_inst_sort=datetime_suite_inst[Q_suite_inst_sort_ind]
                     
## Width-Streamflow Relationship
if width=='regression':
     # Only include for samples collected thorough 1998
    w_regression=w_suite_inst[19:51]
    w_Q_regression=Q_suite_inst[19:51]
     # Only include for flow values above 19 m3/s
    w_Q_threshold=np.where(w_Q_regression>19)
    w_regression=w_regression[w_Q_threshold]
    w_Q_regression=w_Q_regression[w_Q_threshold]
    # Compute regression statistics
    w_1,w_0,w_r, w_p, w_err=scipy.stats.linregress(np.log10(w_Q_regression),np.log10(w_regression))
    a_Wf=10**w_0
    b_Wf=w_1
    w_rsq=w_r**2
    print("r-squared:", w_rsq)
    # Compute geometry for streamflow samples
    Wf_Q=a_Wf*((Q_suite_inst_sort)**b_Wf)
    # Plot
    fig, ax=plt.subplots(1,1,figsize=(6,4))
    plt.plot(w_Q_regression, w_regression,'ro',label='Observed (n=28)',markersize=8,markeredgecolor='black',
               markeredgewidth=1)
    plt.plot(Q_suite_inst_sort, Wf_Q,'b-',label='Fitted Equation',markersize=8,markeredgecolor='black',
               linewidth=3)
    plt.xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
    plt.ylabel('Mean Stream Width (m)',fontsize=12)
    plt.legend()
    plt.title('Mean Stream Width vs. Discharge',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid(which='both')

if width=='constant':
    w_suite_inst=w_inst_all[ind_d_inst]
    Q_suite_inst=Q_inst_all[ind_d_inst]
    datetime_suite_inst=datetime_d_inst
     # Only include for samples collected thorough 1998
    w_regression=w_suite_inst[19:51]
    w_Q_regression=Q_suite_inst[19:51]
     # Only include for flow values above 19 m3/s
    w_Q_threshold=np.where(w_Q_regression>19)
    w_regression=w_regression[w_Q_threshold]
    w_Q_regression=w_Q_regression[w_Q_threshold]
    # Compute regression statistics
    Wf_Q=np.ones(len(Q_suite_inst_sort))*constant_width
    # Plot
    fig, ax=plt.subplots(1,1,figsize=(6,4))
    plt.plot(w_Q_regression, w_regression,'ro',label='Observed (n=28)',markersize=8,markeredgecolor='black',
               markeredgewidth=1)
    plt.plot(Q_suite_inst_sort, Wf_Q,'b-',label='Width=41.6 m',markersize=8,markeredgecolor='black',
               linewidth=3)
    plt.xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
    plt.ylabel('Mean Stream Width (m)',fontsize=12)
    plt.legend()
    plt.title('Mean Stream Width vs. Discharge',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid(which='both')
  
## Depth-Streamflow Relationship
d_inst_sort=d_inst[Q_suite_inst_sort_ind]
d_regression=d_inst
d_Q_regression=Q_suite_inst
 # Only include for samples collected thorough 1998
d_regression=d_regression[19:51]
d_Q_regression=d_Q_regression[19:51]
# Only include for flow values above 19 m3/s
d_Q_threshold=np.where(d_Q_regression>19) 
d_regression=d_regression[d_Q_threshold]
d_Q_regression=d_Q_regression[d_Q_threshold]
# Compute regression statistics
d_1,d_0,d_r, d_p, d_err=scipy.stats.linregress(np.log10(d_Q_regression),np.log10(d_regression))
a_Df=10**d_0
b_Df=d_1
d_rsq=d_r**2
print("r-squared:", d_rsq)
# Compute geometry for streamflow samples
Df_Q=a_Df*((Q_suite_inst_sort)**b_Df)
# Plot
fig, ax=plt.subplots(1,1,figsize=(6,4))
plt.plot(d_Q_regression, d_regression,'ro',label='Observed (n=28)',markersize=8,markeredgecolor='black',
           markeredgewidth=1)
plt.plot(Q_suite_inst_sort, Df_Q,'b-',label='Fitted Equation',markersize=8,markeredgecolor='black',
           linewidth=3)
plt.xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
plt.ylabel('Mean Stream Depth (m)',fontsize=12)
plt.legend(loc=4)
plt.title('Mean Stream Depth vs. Discharge',fontsize=12)
plt.tick_params(labelsize=12)
plt.grid(which='both')
#
## Velocity-Streamflow relationship
u_inst_sort=u_inst[Q_suite_inst_sort_ind]
u_regression=u_inst
u_Q_regression=Q_suite_inst
 # Only include for samples collected thorough 1998
u_regression=u_regression[19:51]
u_Q_regression=u_Q_regression[19:51]
#  Only include for flow values above 19 m3/s
u_Q_threshold=np.where(u_Q_regression>19)
u_regression=u_regression[u_Q_threshold]
u_Q_regression=u_Q_regression[u_Q_threshold]
# Compute regression statistics
u_1, u_0, u_r, u_p, u_err=scipy.stats.linregress(np.log10(u_Q_regression), np.log10(u_regression))
a_Uf=10**u_0
b_Uf=u_1
u_rsq=u_r**2
print("r-squared:", u_rsq)
# Compute geometry for streamflow samples
Uf_Q=a_Uf*((Q_suite_inst_sort)**b_Uf)
# Plot
fig, ax=plt.subplots(1,1,figsize=(6,4))
plt.plot(u_Q_regression, u_regression,'ro', label='Observed (n=28)',markersize=8,markeredgecolor='black',
           markeredgewidth=1)
plt.plot(Q_suite_inst_sort, Uf_Q,'b-',label='Fitted Equation',markersize=8,markeredgecolor='black',
           linewidth=3)
plt.legend(loc='best')
plt.xlabel ('Discharge (m$^{3}$/s)',fontsize=12)
plt.ylabel('Mean Stream Velocity (m/s)',fontsize=12)
plt.tick_params(labelsize=12)
plt.grid(which='both')
plt.title('Mean Stream Velocity vs. Discharge',fontsize=12)
#%% Roughness and Shear Stress
# Total Roughness from Manning's n
# Observations- use same set of observations as hydraulic geometry
Q_n=u_Q_regression
u_n_obs=u_regression
d_n_obs=d_regression
Rh_n_obs=d_n_obs # Assume that Rh=d
nt_obs=(Rh_n_obs**(2/3))*(S**(1/2))/u_n_obs # Manning's n

# Hydraulic Geometry
d_n=a_Df*Q_n**b_Df
u_n=a_Uf*Q_n**b_Uf
Rh_n=d_n
nt=(Rh_n**(2/3))*(S**(1/2))/u_n

# Compute ng0- from Observations
ng_obs_df=compute_ng(d_n_obs, Rh_n_obs, Ch_d50_m, Ch_d65_m, Ch_d84_m, Ch_d90_m, d_n_obs)
# create data frame with 10 n_g equations results with d_n_obs as the index
ng_obs_bar=sum(ng_obs_df['Bray, 1979']*d_n_obs)/(sum(d_n_obs)) # depth-weighted average of ng
ng_obs_df_sort=ng_obs_df.sort_index(axis=0) # sorted data frame
ng_obs_star_df_sort=ng_obs_df_sort/ng_obs_bar # divide by mean

#Plot ng_obs
fig_ng_obs, ax_ng_obs = plt.subplots()
ax1_ng_obs=ax_ng_obs.twinx()
ng_obs_plt=ng_obs_df_sort.plot(ax=ax_ng_obs, marker='.', markersize=3,linewidth=1, 
                         logx=False, logy=False, legend=False)
ng_obs_plt.set_xlabel('H (m)')
ng_obs_plt.set_ylabel('n$_{g}$')
ng_obs_plt.set_title('Grain roughness (n$_{g}$) vs. Depth (H)',fontsize=12)
ng_obs_star_plt=ng_obs_star_df_sort.plot(ax=ax1_ng_obs, marker='.', legend=False)
ng_obs_star_plt.set_title('Grain roughness (n$_{g}$) vs. Depth (H)',fontsize=12)
plt.ylabel('n$_{g*}$', fontsize=12, rotation=-90, labelpad=15)

# Compute the modified n
###########
lower_ntnd=0.6
upper_ntnd=1.25
CI=0.95
###########
a_n, b_n, a_Df_n, b_Df_n, a_Uf_n, b_Uf_n, rsq, nt_obs_nd, ind_n=compute_nt(Q_n, d_n_obs, nt_obs,lower_ntnd, upper_ntnd, CI) # remove high and one low outlier
n_cbev=(a_n*d_n_obs**b_n)*ng_obs_bar
n_cbev_df=pd.DataFrame(data=[n_cbev]).transpose()
n_cbev_df.columns=['Calibrated']
n_cbev_df.index=d_n_obs

#Plot of ng and ng, adj versus H.
ng_obs_df_sort.plot(marker='o', markersize=4,linewidth=1, logx=False, 
                    logy=False, legend=False)
plt.plot(np.sort(d_n_obs), n_cbev[np.argsort(d_n_obs)], 'ko-', 
         label='n$_{g,mod}$ (Equation 11)')
plt.xlabel('H (m)')
plt.ylabel('n$_{g}$ or n$_{g,adj}$')
plt.title('n$_{g}$ or n$_{g,adj}$ versus H',fontsize=12)

# Shear Stress- Steady State, uniform flow
tau_obs=rho_w*g*Rh_n_obs*S

# Grain shear stress
tau_prime_df=ng_obs_df.apply(lambda n: rho_w*g*((n*u_n)**(3/2))*S**(1/4), axis=0)
tau_prime_df.index=ng_obs_df.index
tau_prime_df_sort=tau_prime_df.sort_index(axis=0)

# Adjusted shear stress
n_cbev_Q=ng_obs_bar*(a_Df_n**b_n)*a_n*Q_n**(b_Df_n*b_n)
tau_prime_cbev=rho_w*g*((n_cbev_Q*a_Uf_n*Q_n**b_Uf_n)**(3/2))*S**(1/4)

# Plot
tau_plt=tau_prime_df_sort.plot(marker='o', markersize=4,linewidth=0.5,
                               loglog=False, legend=False)
plt.plot(a_Df*np.sort(Q_n)**b_Df, tau_prime_cbev[np.argsort(Q_n)], 'ko-', 
         label='n$_{g,mod}$ (Equation 11)')

tau_plt.set_xlabel('H (m)')
tau_plt.set_ylabel('tau$_{g}$ or tau$_{g, mod}$ (Pa)')
plt.title('tau$_{g}$ or tau$_{g, mod}$ versus H',fontsize=12)
plt.xlim(0.8,2.4)

#%% Bedload Transoprt: Wilcock & Crowe, 2003
# Compute hydraulic geomwtry
Wf=constant_width*np.ones([BL_sample_n]) # Use constant 41.6- the median width- since poor regression relationship
Uf=a_Uf*BL_sample_wdis**b_Uf
Df=a_Df*BL_sample_wdis**b_Df
Rh=Df
ng=ng_obs_bar*a_n*Df**b_n
tau_BL_obs=rho_w*g*((ng*Uf)**(3/2))*S**(1/4)

# Pre-tau adjustment  (ng bray)
n_bray_compare=(1/(np.sqrt(8*g)))*(Rh**(1/6))/(1.26-2.16*np.log10(Ch_d90_m/Rh))  
tau_BL_compare=rho_w*g*((n_bray_compare*Uf)**(3/2))*S**(1/4)

# All Fractions-- pre-tau adjustment 
## Observed
Wstar_obs_c, Wstar_tot_obs_c, dis_m2s_tot_obs_c, dis_m3s_BL_obs_c=run_wc2003_all_obs (BLopng_inst, BL_sample_dis, tau_BL_compare, BL_sample_Fall, Wf)
## Modeled
phi_line_c, Wstar_line_c, phi_WC_c, Wstar_tot_WC_c, dis_m2s_tot_WC_c=run_wc2003_all_model (Ch_Fs, Ch_Dmean_m, Chopng_inst, tau_BL_compare)
## Error Metrics
NSE, r2=compute_NSE_rs (dis_m2s_tot_WC_c, dis_m2s_tot_obs_c)


# Two Fractions-- pre-tau adjustment 
## Observed
Wstar_gravel_obs_c, Wstar_sand_obs_c, dis_m2s_gravel_obs_c, dis_m2s_sand_obs_c, dis_m2s_BL_obs_c, dis_m3s_BL_obs_c=run_wc2003_2F_obs (BL_sample_dis, BL_sample_Fg, BL_sample_Fs, tau_BL_compare, Wf)            
## Modeled
phi2_line_c, Wstar2_line_c, phi_gravel_W2_c, phi_sand_W2_c, Wstar_gravel_W2_c, Wstar_sand_W2_c, dis_m2s_gravel_W2_c, dis_m2s_sand_W2_c, dis_m2s_total_W2_c=run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_BL_compare)
## Error Metrics
NSE, r2=compute_NSE_rs (dis_m2s_total_W2_c, dis_m2s_BL_obs_c)
print ('NSE- 2F, pre-tau=',NSE)
print ('daily r2- 2F, pre-tau=',r2)

# All Fractions
## Observed
Wstar_obs, Wstar_tot_obs, dis_m2s_tot_obs, dis_m3s_BL_obs=run_wc2003_all_obs (BLopng_inst, BL_sample_dis, tau_BL_obs, BL_sample_Fall, Wf)
## Modeled
phi_line, Wstar_line, phi_WC, Wstar_tot_WC, dis_m2s_tot_WC=run_wc2003_all_model (Ch_Fs, Ch_Dmean_m, Chopng_inst, tau_BL_obs)
## Error Metrics
NSE, r2=compute_NSE_rs (dis_m2s_tot_WC, dis_m2s_tot_obs)

# Two Fractions
## Observed
Wstar_gravel_obs, Wstar_sand_obs, dis_m2s_gravel_obs, dis_m2s_sand_obs, dis_m2s_BL_obs, dis_m3s_BL_obs=run_wc2003_2F_obs (BL_sample_dis, BL_sample_Fg, BL_sample_Fs, tau_BL_obs, Wf)            
## Modeled
phi2_line, Wstar2_line, phi_gravel_W2, phi_sand_W2, Wstar_gravel_W2, Wstar_sand_W2, dis_m2s_gravel_W2, dis_m2s_sand_W2, dis_m2s_total_W2=run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_BL_obs)
## Error Metrics
NSE, r2=compute_NSE_rs (dis_m2s_total_W2, dis_m2s_BL_obs)
print ('NSE- 2F, post-tau=',NSE)
print ('daily r2- 2F, post-tau=',r2)
#%% Compare results before and after tau adjustment
# Plots
# Calibration Plot
i=1
fig1, (ax1, ax2)=plt.subplots(1,2,figsize=(8,8), sharey=True)

# Sand Obs
ax1.loglog(phi_sand_W2_c, Wstar_sand_obs_c,'tab:gray',marker='o',linestyle='None',markersize=6,
           markeredgewidth=1)
ax1.loglog(phi_sand_W2_c[i], Wstar_sand_obs_c[i],'tab:gray',marker='o',linestyle='None',markersize=6,
           markeredgewidth=1, label='Observed- Sand, with n$_{g,b}$')
# Gravel  Obs    
ax1.loglog(phi_gravel_W2_c, Wstar_gravel_obs_c,'tab:gray',marker='s',linestyle='None', markersize=6,
           markeredgewidth=1)
ax1.loglog(phi_gravel_W2_c[i], Wstar_gravel_obs_c[i],'tab:gray',marker='s',linestyle='None',markersize=6,
           markeredgewidth=1, label='Observed- Gravel, with n$_{g,b}$')

# Sand Obs
ax1.loglog(phi_sand_W2, Wstar_sand_obs,'bo',markersize=6,markeredgecolor='black',
           markeredgewidth=1)
ax1.loglog(phi_sand_W2[i], Wstar_sand_obs[i],'bo',markersize=6,markeredgecolor='black',
           markeredgewidth=1, label='Observed- Sand, with n$_{g,E}$')
# Gravel  Obs    
ax1.loglog(phi_gravel_W2, Wstar_gravel_obs,'ys',markersize=6,markeredgecolor='black',
           markeredgewidth=1)
ax1.loglog(phi_gravel_W2[i], Wstar_gravel_obs[i],'ys',markersize=6,markeredgecolor='black',
           markeredgewidth=1, label='Observed- Gravel with n$_{g,E}$')
# Line
ax1.loglog(phi2_line, Wstar2_line,'k--', markersize=5, markeredgecolor='black',
           markeredgewidth=1, linewidth=3, label='Modeled')

ax1.set_ylim([10**-3, 10**1])
ax1.set_xlim([10**-0.5, 10**1.5])
ax1.legend(loc='best')
ax1.set_xlabel("$\phi $")
ax1.set_ylabel("W*$_{i}$",fontsize=12)
ax1.tick_params(labelsize=12, labelleft='on')
ax1.grid(which='both')
ax1.set_title('Calibration Curve- Wilcock and Crowe (2003) \nModified Two-Fraction Form',fontsize=12)  
#%%
# All Fraction
ax2.loglog(phi_WC_c.values,Wstar_obs_c.values,'tab:gray',marker='o',
   markersize=2, linestyle='None', markeredgewidth=1);
ax2.loglog(phi_WC_c.values[0][0],Wstar_obs_c.values[0][0],'tab:gray',
   marker='o',markersize=2, linestyle='None', markeredgewidth=1, label='Observed, with n$_{b}');

ax2.loglog(phi_WC.values,Wstar_obs.values,'r.', markeredgecolor='black',
   markeredgewidth=1);
ax2.loglog(phi_WC.values[0][0],Wstar_obs.values[0][0],'r.', markeredgecolor='black',
   markeredgewidth=1, label='Observed, with n$_{E}');
ax2.loglog(phi_line, Wstar_line,'k--',markersize=8, markeredgecolor='black',
   markeredgewidth=1, linewidth=3,label='Modeled');
          
ax2.set_ylim([10**-3, 10**1])
ax2.set_xlim([10**-0.5, 10**1.5])
ax2.legend(loc='best')
ax2.set_xlabel("$\phi $")
ax2.tick_params(labelsize=12, labelleft='on')
ax2.grid(which='both')
ax2.set_title('Calibration Curve \n Wilcock and Crowe (2003) Model',fontsize=12)  

# One-to-one Plot
plt.figure()
plt.loglog([10**-8, 10**-2], [10**-8, 10**-2],'k-')
plt.loglog([10**-8, 10**-3], [10**-7, 10**-2],'k--')
plt.loglog([10**-7, 10**-2], [10**-8, 10**-3],'k--')
plt.ylim([10**-7, 10**-2])
plt.xlim([10**-7, 10**-2])

plt.loglog(dis_m2s_total_W2_c, dis_m2s_BL_obs_c, 'tab:gray',marker='s',linestyle='None',
           markersize=6,markeredgecolor='None',
           markeredgewidth=1,label='With n$_{g,b}$')

plt.loglog(dis_m2s_total_W2, dis_m2s_BL_obs, 'gs',markersize=6,markeredgecolor='black',
           markeredgewidth=1,label='With n$_{g,E}$')

plt.ylabel("Observed Discharge (m$^{2}$/s)",fontsize=12)
plt.xlabel("Modeled Discharge (m$^{2}$/s)",fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.tick_params(labelsize=12)
plt.grid(which='both')
plt.title('Bedload Discharge-Observed vs. Modeled\n Wilcock and Crowe (2003) Modified Two-Fraction Form', fontsize=12)

#%% Compare two-fraction and all-fraction
# Plots
# Calibration Plot
i=1
fig1, (ax1, ax2)=plt.subplots(1,2,figsize=(8,8), sharey=True)

# Sand Obs
ax1.loglog(phi_sand_W2, Wstar_sand_obs,'bo',markersize=6,markeredgecolor='black',
           markeredgewidth=1)
ax1.loglog(phi_sand_W2[i], Wstar_sand_obs[i],'bo',markersize=6,markeredgecolor='black',
           markeredgewidth=1, label='Observed- Sand')
# Gravel  Obs    
ax1.loglog(phi_gravel_W2, Wstar_gravel_obs,'ys',markersize=6,markeredgecolor='black',
           markeredgewidth=1)
ax1.loglog(phi_gravel_W2[i], Wstar_gravel_obs[i],'ys',markersize=6,markeredgecolor='black',
           markeredgewidth=1, label='Observed- Gravel')
# Line
ax1.loglog(phi2_line, Wstar2_line,'k--', markersize=5, markeredgecolor='black',
           markeredgewidth=1, linewidth=3, label='Modeled')

ax1.set_ylim([10**-3, 10**1])
ax1.set_xlim([10**-0.5, 10**1.5])
ax1.legend(loc='best')
ax1.set_xlabel("$\phi $")
ax1.set_ylabel("W*$_{i}$",fontsize=12)
ax1.tick_params(labelsize=12, labelleft='on')
ax1.grid(which='both')
ax1.set_title('Two-Fraction',fontsize=12)  

# All Fraction
ax2.loglog(phi_WC.values,Wstar_obs.values,'r.', markeredgecolor='black',
   markeredgewidth=1);
ax2.loglog(phi_WC.values[0][0],Wstar_obs.values[0][0],'r.', markeredgecolor='black',
   markeredgewidth=1, label='Observed');
ax2.loglog(phi_line, Wstar_line,'k--',markersize=8, markeredgecolor='black',
   markeredgewidth=1, linewidth=3,label='Modeled');
          

ax2.set_ylim([10**-3, 10**1])
ax2.set_xlim([10**-0.5, 10**1.5])
ax2.legend(loc='best')
ax2.set_xlabel("$\phi $")
ax2.tick_params(labelsize=12, labelleft='on')
ax2.grid(which='both')
ax2.set_title('All Fractions',fontsize=12)  

# One-to-one Plot
plt.figure()
plt.loglog([10**-8, 10**-2], [10**-8, 10**-2],'k-')
plt.loglog([10**-8, 10**-3], [10**-7, 10**-2],'k--')
plt.loglog([10**-7, 10**-2], [10**-8, 10**-3],'k--')
plt.ylim([10**-7, 10**-4])
plt.xlim([10**-7, 10**-4])


plt.plot(dis_m2s_tot_WC, dis_m2s_tot_obs, 'k^',markersize=6,markeredgecolor='black',
           markeredgewidth=1,label='All Fractions')
plt.loglog(dis_m2s_total_W2, dis_m2s_BL_obs, 'gs',markersize=6,markeredgecolor='black',
           markeredgewidth=1,label='Two-Fraction')
plt.ylabel("Observed Discharge (m$^{2}$/s)",fontsize=12)
plt.xlabel("Modeled Discharge (m$^{2}$/s)",fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.tick_params(labelsize=12)
plt.grid(which='both')
plt.title('Wilcock and Crowe (2003) Bedload Discharge\nObserved vs. Modeled', fontsize=12)

#%% Suspended Sediment
# Develop relationship for percent silt versus streamflow
pcnt_silt=SSdia_inst[:,5]                               # percent of silt in each sample
q_SS=q_SSdia_inst                                       # streamflow when sample collected
SSC_when_dist=SSC_inst_all[ind_SSdia_inst]              # ssc when distritbution sample collected
pcnt_silt_avg=np.sum(pcnt_silt*q_SS)/(100*np.sum(q_SS)) # flow weighted average percent silt
ind_ss_pcnt=(pcnt_silt>60)&(q_SS>=30)&(q_SS<200)        # index for ss percent to be analyzed

# power regression for percent silt
ps_1, ps_0, ps_r, ps_p, ps_err=scipy.stats.linregress(np.log10(q_SS[ind_ss_pcnt]),
                                                      np.log10(pcnt_silt[ind_ss_pcnt]))
a_ps=10**ps_0
b_ps=ps_1
ps_reg=a_ps*(np.sort(q_SS[ind_ss_pcnt])**b_ps)                             

plt.figure()
plt.semilogx(q_SS, pcnt_silt,color='grey',marker='o', linestyle='none', label='Suspended sediment samples- Excluded from regression')
plt.semilogx(q_SS[ind_ss_pcnt], pcnt_silt[ind_ss_pcnt], 'bo', label='Suspended sediment samples- Used in regression')
plt.semilogx(np.sort(q_SS[ind_ss_pcnt]), ps_reg, 'k-', label='Empirical relationship for percent silt versus streamflow')
plt.semilogx([np.min(q_SS),np.min(q_SS[ind_ss_pcnt])], [83.5, 83.5], 'k-')
plt.semilogx([np.max(q_SS),np.max(q_SS[ind_ss_pcnt])], [59, 59], 'k-')
plt.xlabel('Flow (cms)')
plt.ylabel('Percent Silt')
plt.title('Percent Silt vs. Streamflow')
plt.legend()

print('Min flow:', np.min(q_SS[ind_ss_pcnt]))
print('Max flow:', np.max(q_SS[ind_ss_pcnt]))

print('Min percent:', np.min(ps_reg))
print('Max percent:', np.max(ps_reg))

#Check Suspended Sediment Rating Curve from Curran et al., 2009
# load data
os.chdir(datadir)
q_ssc=np.load('Q_SSC_regression.npy')
ssc_mgL=np.load('SSC_regression.npy')    
dis_ss_obs_m3s=ssc_mgL*q_ssc/(sg*1000*1000)

# Suspended Sediment Rating Curve- Check!
# Compute channel geometry
W_ss, U_ss, D_ss, Rh_ss, tau_ss=compute_channel_geom(constant_width, a_Uf, b_Uf,a_Df, b_Df, a_n, b_n, ng_obs_bar, q_ssc, S)
# Compute suspended sediment
dis_ss_m2s, dis_ss_m3s=compute_SS_RC (a_s, b_s_c, b_s_l, cf_s, K, q_ssc, W_ss)
    
# One-to-one Plot
plt.figure()
plt.loglog([10**-7, 10**0], [10**-7, 10**0],'k-')
plt.loglog([10**-7, 10**-1], [10**-6, 10**0],'k--')
plt.loglog([10**-6, 10**0], [10**-7, 10**-1],'k--')
plt.ylim([10**-7, 10**-1])
plt.xlim([10**-7, 10**-1])

plt.loglog(dis_ss_m2s, dis_ss_obs_m3s/W_ss, 'gs',markersize=8,
           markeredgecolor='black', markeredgewidth=1, label='Rating Curve')
plt.xlabel("Modeled Discharge (m$^{2}$/s)",fontsize=12)
plt.ylabel("Observed Discharge (m$^{2}$/s)",fontsize=12)
plt.title ('Suspended Sediment Rating Curve')
plt.tick_params(labelsize=12)
plt.grid(which='both')

#%% Rating Curve for Curran et al. (2009)
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
Q_s_erosion=1000*Q_s_Vsed_totann_LM/A_LM     # Annual ss eroded from hillslopes (mm/year)
## Distinguish between silt and sand in reservoir
Q_s_Vsed_totann_silt_GC=Q_s_totann_silt/bd_fine # Annual silt in reservoir- before being transported out
Q_s_Vsed_totann_sand_GC=Q_s_totann_sand/bd_coarse # Annual sand in reservoir
Q_s_Vsed_totann_GC=Q_s_Vsed_totann_silt_GC+Q_s_Vsed_totann_sand_GC# Annual SS contribution at Glines Canyon dam (m3/yr)
## Incorporate reservir trap efficiency, assuming only silt flows over (not sand)
Q_s_Vsed_totann_silt_GC=Q_s_Vsed_totann_silt_GC-Q_s_Vsed_totann_GC*(1-rte)
Q_s_Vsed_totann_GC=Q_s_Vsed_totann_silt_GC+Q_s_Vsed_totann_sand_GC

# Bedload
Q_b_percentile=cf_b*a_b*Q_RC**b_b             # Bedload (Mg/d) for each percentile of interest at LM
Q_b_annual_contribution=Q_b_percentile*dp*365.25      # Annual contribution of bedload (Mg/yr) for each percentile of interest at LM
Q_b_annual_total=np.sum([Q_b_annual_contribution])    # Annual bedload contribution (Mg/yr) at LM
Q_b_Vsed_annual_total_LM=Q_b_annual_total/sg          # Annual bedload passing Lake Mills gage (m3/yr)- since using specific gravity
Q_b_erosion=1000*Q_b_Vsed_annual_total_LM/A_LM        # Annual bedload eroded from watershed (mm/year)
Q_b_Vsed_annual_total_GC=Q_b_annual_total/bd_coarse   # Annual bedload contribution at Glines Canyon dam (m3/yr)- since using bulk density
Q_b_Vsed_total_GC=Q_b_Vsed_annual_total_GC*n_years_GC # Total bedload at Glines Canyon Dam (m3)

# Computed dam sedimentation rate, incorporating trap efficiency of reservoir with suspended sediment
Q_t_Vsed_annual_total_GC=Q_s_Vsed_totann_GC+Q_b_Vsed_annual_total_GC  # total sedmentation rate per year
Q_t_Vsed_lifetime_total=Q_t_Vsed_annual_total_GC*n_years_GC        # Estimate sediment accumulation over dam lifetime (m3)
Q_b_Vsed_pcnt_GC=Q_b_Vsed_annual_total_GC/Q_t_Vsed_annual_total_GC # percent of bedload in dam  (m3)
Q_s_Vsed_pcnt_GC=Q_s_Vsed_totann_GC/Q_t_Vsed_annual_total_GC        # percent of SS in dam (m3)

Q_t_lifetime_dif=Q_t_Vsed_lifetime_total-Vsed_GC    # Difference between modeled and estimated sediment accumulation (m3)
Q_t_lifetime_pctdif=100*Q_t_lifetime_dif/Vsed_GC    # percent difference

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
phi_line_RC, Wstar_line_RC, phi_gravel_RC, phi_sand_RC, Wstar_gravel_RC, Wstar_sand_RC, dis_m2s_gravel_RC, dis_m2s_sand_RC, dis_m2s_total_RC=run_wc2003_2F_model (Ch_Fs, Ch_Dmean_m, Ch_Dsand_m, Ch_Dgravel_m, tau_RC)
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
       
#%% Plot Rating Curve
fig1, ax1=plt.subplots(1,1,figsize=(4,3))
ax2=ax1.twinx()
ax2.semilogy(1-cum_pcntl, Q_RC*3600*24,'k.',markersize=3, label='Streamflow')
ax1.semilogy(1-cum_pcntl, dis_m3s_suspended_RC*3600*24,'r.',markersize=3, label='Suspended Load- USGS')
#ax1.semilogy(1-cum_pcntl, dis_m3s_bedload_RC*3600*24,'g.',markersize=3, label='Bedload - USGS')
ax1.semilogy(1-cum_pcntl, dis_m2s_total_RC*3600*24*W_RC,'b.',markersize=3, label='Bedload - W&C-2F, ng,E')
#ax1.semilogy(1-cum_pcntl, dis_m2s_total_RC_pre*3600*24*W_RC,'c.',markersize=3, label='Bedload - W&C-2F, ng,b')

ax1.set_xlabel('Fraction of time discharge equaled or exceeded', fontsize=12)
ax1.set_ylabel('Sediment Discharge (m$^{3}$/day)', fontsize=12)
ax2.set_ylabel('Streamflow Discharge (m$^{3}$/day)', fontsize=12)
#ax1.set_ylim(10**-5,10**4)s
#ax2.set_ylim([10**0, 10**3])
ax1.legend(loc=3)
ax2.legend(loc=4)
ax1.grid(which='major', axis='x')
#plt.title('Daily Sediment and Streamflow Discharge \n Duration Curves at Lake Mills gage',fontsize=12)
plt.tick_params(labelsize=12)

#%% Compute reservoir sedimentation volumes
# Total reservoir sedimentation (m3 over 84 years)  
WC_res_life_m3=WC_ann_res_m3yr*n_years_GC
WC_gravel_res_life_m3=WC_gravel_ann_res_m3yr*n_years_GC
WC_sand_res_life_m3=WC_sand_ann_res_m3yr*n_years_GC

WC_res_life_m3_pre=WC_ann_res_m3yr_pre*n_years_GC
WC_gravel_res_life_m3_pre=WC_gravel_ann_res_m3yr_pre*n_years_GC
WC_sand_res_life_m3_pre=WC_sand_ann_res_m3yr_pre*n_years_GC

BL_res_life=Q_b_Vsed_annual_total_GC*n_years_GC
SS_sand_res_life_m3=Q_s_Vsed_totann_sand_GC*n_years_GC
SS_silt_res_life_m3=Q_s_Vsed_totann_silt_GC*n_years_GC
   
print('Lifetime reservoir sedimentation (10^6 m3)')
print('Bedload Rating Curve:',BL_res_life/10**6)
print('Wilcock & Crowe:', WC_res_life_m3/10**6)
print('Wilcock & Crowe- Gravel:', WC_gravel_res_life_m3/10**6)
print('Wilcock & Crowe- Sand:', WC_sand_res_life_m3/10**6)

print('Wilcock & Crowe- pre:', WC_res_life_m3_pre/10**6)
print('Wilcock & Crowe- Gravel- pre:', WC_gravel_res_life_m3_pre/10**6)
print('Wilcock & Crowe- Sand- pre:', WC_sand_res_life_m3_pre/10**6)

print('Suspended Load Rating Curve:', (SS_sand_res_life_m3+SS_silt_res_life_m3)/10**6)
print('Suspended Load Rating Curve- Sand:', SS_sand_res_life_m3/10**6)
print('Suspended Load Rating Curve- Silt:', SS_silt_res_life_m3/10**6)