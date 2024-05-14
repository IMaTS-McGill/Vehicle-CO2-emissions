import numpy as np
import pandas as pd
import math
from datetime import timedelta
from geopy import distance
from pyproj import Transformer, CRS
from numpy import ma

def duplicate_index_correction(df_in):
    '''
    Check for duplicated index and return DataFrame with duplicated values removed.
    
    If there are duplicated DateTime, checks for missing DateTime 1 sec before or after.
    Change one of the duplicated indexes to the missing index.
    '''
    df = df_in.copy()
    df_dup = df.index[df.index.duplicated(False)]
    for t in df_dup:
        ix = np.where(df.index == t)[0]
        if (t + timedelta(seconds=1)) in df.index and (t - timedelta(seconds=1)) not in df.index:
            df.index.values[ix[0]] = (t - timedelta(seconds=1))
        elif (t + timedelta(seconds=1)) not in df.index and (t - timedelta(seconds=1)) in df.index:
            df.index.values[ix[1]] = (t + timedelta(seconds=1))
            
    return df

def subset_and_rename(df_in, names):
    '''
    Subset DataFrame with raw variable names and rename with processed variable names.
    '''
    df = df_in.copy()
    names_dict = dict(zip(names['raw_name'], names['processed_name']))
    subset_dict = dict((k, names_dict[k]) for k in names_dict.keys() if k in df.columns)
    subset_df = df[subset_dict.keys()]
    
    return subset_df.rename(columns=subset_dict)

def find_start_end(df_in, variable_names_pems, variable_names_obd):
    '''
    Return first and last valid indexes.
    '''
    var_names = variable_names_pems['processed_name'].to_list()
    var_names.remove('DateTime')
    var_names.remove('Bag#')
    first_pems = df_in[var_names].first_valid_index()
    last_pems = df_in[var_names].last_valid_index()

    var_names = variable_names_obd['processed_name'].to_list()
    var_names.remove('DateTime')
    first_obd = df_in[var_names].first_valid_index()
    last_obd = df_in[var_names].last_valid_index()

    start_with = max(first_pems, first_obd)
    end_with = min(last_pems, last_obd)
    
    return (start_with, end_with)

def discard_long_stops(df_in, discard_threshold):
    '''
    Remove rows with consecutive stopping longer than 'discard_threshold' from DataFrame. 
    '''
    print('Before long stop removal data dimensions are :', df_in.shape[0])
    df = df_in.copy()
    n = df.shape[0]
    zero_speed_diff = (df['GPS Speed [km.h-1]'] == 0) & ( (df['GPS Speed [km.h-1]'].diff() == 0) | (df['GPS Speed [km.h-1]'].diff().isna()) )
    l = len(zero_speed_diff)
    stopping = np.empty((l, 2))
    stopping[:] = np.nan
    c = 0
    for i in range(0, l):
        flag = False
        if i == 0 and zero_speed_diff[0]:
            stopping[c, 0] = 0
            flag = True
        elif not zero_speed_diff[i-1] and zero_speed_diff[i] :
            stopping[c, 0] = i
            flag = True
        if flag:
            for j in range((i+1), l):
                if zero_speed_diff[j-1] and not zero_speed_diff[j]:
                    stopping[c, 1] = j
                    c = c + 1
                    break

    stopping = stopping[~np.isnan(stopping).any(axis=1)]
    stopping = pd.DataFrame(stopping)
    stopping['diff'] = (df.index[stopping[1].astype(int)] - df.index[stopping[0].astype(int)]).total_seconds()
    discard_datetimes = stopping[stopping['diff'] > discard_threshold]
    
    discard_ix = []
    for i in range(0, discard_datetimes.shape[0]):
        (s, e) = discard_datetimes.iloc[i, [0,1]]
        discard_ix.append(df.index[int(s+15):int(e-15)])
    for ix in discard_ix:
        df.drop(ix, axis=0, inplace=True)

    print('After long stop removal data dimensions are :', df.shape[0])
    
    return df

def fill_missing_rows(df_in):
    '''
    Impute missing values of a row with the values of neigbour rows.
    Note: Only if there is a 1-second missing reading.
    '''
    df = df_in.copy()
    date_range = pd.date_range(df.index.min(), df.index.max(), freq='S')
    df = df.reindex(date_range)
    ix = df.isna().sum(axis=1)
    ix_na = ix[ix > 0].index
    for i in ix_na:
        prev_i = i - timedelta(seconds=1)
        next_i = i + timedelta(seconds=1)
        if prev_i in df.index and next_i in df.index:
            if ix[prev_i] == 0 and ix[next_i] == 0:
                for c in df.columns[df.loc[i].isna()]:
                    df.loc[i, c] = (df.loc[prev_i, c] + df.loc[next_i, c])/2
    df = df.dropna(axis=0)
    if (sum(df.isna().sum(axis=1)) == 0):
        print('After missing value imputation data dimensions are :', df.shape)
    else:
        raise Exception('There are rows with missing values.')
        
    return df

def interpolate_gps_coordinates(df_in):
    '''
    Detect instances of frozen GPS measurements, 
    interpolate based on vehicle speed as a weighting profile,
    and compute GPS speed using the interpolated GPS coordinates.
    '''
    df = df_in.copy()
    n = df.shape[0]

    ## combine sequential differences into one dataframe
    s1 = range(-1, n-1)
    s2 = range(0, n)
    diff_seq = pd.DataFrame({'start' : s1, 'end' : s2, 
                             'diff_lat' : df['Latitude [deg]'].diff(),
                             'diff_lon' : df['Longitude [deg]'].diff(),
                             'diff_alt' : df['Altitude [m]'].diff(),
                             'gps_speed' : df['GPS Speed [km.h-1]'],
                             'diff_speed' : df['GPS Speed [km.h-1]'].diff() * 1000 / 3600, 
                             'wheel_speed' : df['Vehicle Speed [km.h-1]'], 
                             'lat_deg' : df['Latitude [deg]'], 
                             'lon_deg' : df['Longitude [deg]'], 
                             'alt' : df['Altitude [m]']})
    diff_seq.reset_index(inplace=True)   
    diff_seq.rename(columns={'index':'DateTime'}, inplace=True)

    ## find the groups of consecutive points where 
    ## GPS coordinates are frozen  
    ## OR 
    ## GPS coordinates are changing but GPS speed is frozen
    diff_seq = diff_seq[ ((diff_seq['diff_lat'] == 0) & (diff_seq['diff_lon'] == 0) & (diff_seq['diff_alt'] == 0)) | (((diff_seq['diff_lat'] != 0) | (diff_seq['diff_lon'] != 0) | (diff_seq['diff_alt'] != 0)) & (diff_seq['gps_speed'] == 0)) ]
    sec = pd.Timedelta('1sec')
    in_block = ((diff_seq['DateTime'] - diff_seq['DateTime'].shift(-1)).abs() == sec) | (diff_seq['DateTime'].diff() == sec)
    filt = diff_seq.loc[in_block]
    breaks = filt['DateTime'].diff() != sec
    groups = breaks.cumsum()
    
    df2 = df[['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]', 'Vehicle Speed [km.h-1]', 'GPS Speed [km.h-1]']].copy()    
    df2.reset_index(inplace=True)
    df2.rename(columns={'index':'DateTime'}, inplace=True)
    df2.set_index('DateTime', inplace=True)

    transformer = Transformer.from_crs(4326, 32188)
    back_transformer = Transformer.from_crs(32188, 4326)
    lat_32188, lon_32188 = transformer.transform(df2['Latitude [deg]'], df2['Longitude [deg]'])
    df2['Latitude [32188]'] = lat_32188
    df2['Longitude [32188]'] = lon_32188

    for _, frame in filt.groupby(groups):
        ix = frame['DateTime'].to_list()
        ix_start = ix[0] - timedelta(seconds=1)
        ix_end = ix[-1] + timedelta(seconds=1)
        if ix_start not in df.index:
            ix_start = ix[0]
        if ix_end not in df.index:
            ix_end = ix[-1]
        ix_bound = [ix_start, ix_end]
        if df2.loc[ix_start:ix_end, 'Vehicle Speed [km.h-1]'].sum() < 10:
            continue
            
        df3 = df2.loc[ix_bound, ['Latitude [32188]', 'Longitude [32188]', 'Altitude [m]']]
        date_range = pd.date_range(df3.index.min(), df3.index.max(), freq='S')
        df3 = df3.reindex(date_range)
        
        w = df2.loc[ix_start:ix_end, 'Vehicle Speed [km.h-1]'] + 0.00001
    
        for c in ['Latitude [32188]', 'Longitude [32188]', 'Altitude [m]']:
            s = df3[c]
            sb = s.fillna(method='ffill')
            se = s.fillna(method='bfill')
            cw = w.cumsum()
            w2 = pd.Series(None, index=s.index)
            w2[~np.isnan(s)] = cw[~np.isnan(s)]
            wb = w2.fillna(method='ffill')
            we = w2.fillna(method='bfill')
            cw = (cw - wb) / (we - wb)
            r = sb + cw * (se - sb)
            r.update(s)
            df3[c] = r
        
        lat_4236, lon_4236 = back_transformer.transform(df3['Latitude [32188]'], df3['Longitude [32188]'])
        df.loc[df3.index, 'Latitude [deg]'] = lat_4236
        df.loc[df3.index, 'Longitude [deg]'] = lon_4236
        df.loc[df3.index, 'Altitude [m]'] = df3['Altitude [m]']
        
        ## distance between sequential pairs
        xyz = np.asarray(df3[['Longitude [32188]', 'Latitude [32188]', 'Altitude [m]']])
        dist_seq = np.empty(xyz.shape[0])
        dist_seq[:] = np.nan
        for counter in range(xyz.shape[0] - 1):
            dist_seq[counter+1] = np.sqrt(np.sum((xyz[counter+1] - xyz[counter])**2))  
        
        ## difference between times
        df3['DateTime'] = df3.index
        diff_time_seq = df3['DateTime'].diff().dt.total_seconds()
        
        ## compute gps speed between sequential pairs
        gps_speed_seq = (dist_seq/diff_time_seq) * 3600 / 1000
        df.loc[df3.index[1:(df3.shape[0]-1)], 'GPS Speed [km.h-1]'] = gps_speed_seq[1:(df3.shape[0]-1)]
        
    return df

def compute_gps_speed(df_in):
    '''
    Compute GPS speed using GPS speed values of the neighbour rows.
    '''
    df = df_in.copy()
    n = df.shape[0]
    
    df2 = df[['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]', 'Vehicle Speed [km.h-1]', 'GPS Speed [km.h-1]']]  
    
    transformer = Transformer.from_crs(4326, 32188)
    back_transformer = Transformer.from_crs(32188, 4326)
    lat_32188, lon_32188 = transformer.transform(df2['Latitude [deg]'], df2['Longitude [deg]'])
    df2['Latitude [32188]'] = lat_32188
    df2['Longitude [32188]'] = lon_32188

    mask = (df2['GPS Speed [km.h-1]'] < 0) | (df2['GPS Speed [km.h-1]'] > 140)
    
    for ix in df2[mask].index:
        
        ix_start = ix - timedelta(seconds = 5)
        ix_end = ix + timedelta(seconds = 5)
        th = 5
        while ix_start not in df2.index and th > 0:
            th -= 1
            ix_start = ix - timedelta(seconds = th)
        th = 5
        while ix_end not in df2.index and th > 0:
            th -= 1
            ix_end = ix + timedelta(seconds = th)
        
        ## distance between neighbours
        xyz1 = np.asarray(df2.loc[ix_start, ['Latitude [32188]', 'Longitude [32188]', 'Altitude [m]']])
        xyz2 = np.asarray(df2.loc[ix_end, ['Latitude [32188]', 'Longitude [32188]', 'Altitude [m]']])
        dist = np.sqrt(np.sum((xyz1 - xyz2)**2))  
        
        ## difference between times
        df3 = df2.loc[[ix_start, ix_end], :]
        df3['DateTime'] = df3.loc[[ix_start, ix_end], :].index
        diff_time = df3['DateTime'].diff().dt.total_seconds()
        
        ## compute gps speed between sequential pairs
        gps_speed = (dist/diff_time[1]) * 3600 / 1000
        df.loc[ix, 'GPS Speed [km.h-1]'] = gps_speed
        
    return df

def compute_slope_geodesic_avg(df_in, window_size = 5, threshold = 0.15):
    '''
    Compute slope using GPS coordinates as the sum of sequential altitude differences of the measurements, 
    divided by the sum of sequential horizontal movement differences (using latitude and longitude) of the measurements.
    '''
    df = df_in.copy()

    ## get the unique lon-lat-alt
    df.drop_duplicates(subset = ['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]'], keep='first', inplace=True)
    n = df.shape[0]
    
    ## horizontal distance between sequential pairs
    yx = df[['Latitude [deg]', 'Longitude [deg]']]
    yx = yx.apply(tuple, axis=1)
    hor_dist_seq = np.empty(n)
    hor_dist_seq[:] = np.nan
    for counter in range(len(yx) - 1):
        # print(df.index[counter], yx[counter], yx[counter + 1])
        hor_dist_seq[counter+1] = distance.distance(yx[counter], yx[counter + 1]).m

    ## vertical distance between sequential pairs
    vert_dist_seq = df['Altitude [m]'].diff()
    
    ## compute slope between sequential pairs
    s1 = range(-1, n-1)
    s2 = range(0, n)
    slope_seq = vert_dist_seq/hor_dist_seq
    slope_seq = np.where(slope_seq > threshold, np.nan, slope_seq)
    slope_seq = np.where(slope_seq < -1*threshold, np.nan, slope_seq)
    slope_seq = pd.Series(slope_seq)
    slope_seq.replace([np.inf, -np.inf], np.nan, inplace=True)
    slope_seq = slope_seq.bfill().ffill()
    slope_seq = pd.DataFrame({'start' : s1, 'end' : s2, 'd' : slope_seq})
    slope_seq.reset_index(inplace=True)    
    
    ## mean slopes for window of t
    b = math.ceil(window_size/2)
    e = window_size - b
    s = range(b, n-e)
    slope = []
    for t in s:
        slope.append(slope_seq.loc[(slope_seq['start'] >= t-b) & (slope_seq['end'] <= t+e), 'd'].mean())
    slope = np.array(slope)
    
    slope_out = np.empty(n)
    slope_out[:] = np.nan
    slope_out[s] = np.array(slope)
    
    df1 = df[['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]']]
    df1['Slope [%]'] = slope_out

    df2 = df_in.copy()
    df_out = pd.merge(df2, df1, on=['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]'])
    
    return np.array(df_out['Slope [%]'])

def neg_emissions_to_zero(df_in):
    '''
    Replace negative emissions with zero.
    '''
    df = df_in.copy()
    columns = ['CO2 Conc. [%]']
    for c in columns:
        if c in df.columns:
            df.loc[df[c] < 0, c] = 0
    return df


def fuel_rate1(fc_in, gasoline_density):
    '''
    Compute fuel rate using gasoline density.
    
    Fuel Rate [g.s-1] = Fuel Rate [l.h-1] * (Gasoline Density [g.l-1] / 3600)
    Gasoline Density = 718~780 [kg.m-3], which depends on the type of gasoline (whether it is ethanol-based or not, additives, etc.)
    
    Input: 
        fc_in -- Fuel Rate [l.h-1]
        
    Output:
        fc_out -- Fuel Rate [g.s-1]
    '''
    fc_out = fc_in * gasoline_density / 3600
    
    return fc_out

def fuel_rate2(maf, phi):
    '''
    Compute fuel rate using mass air flow rate.
    
    Fuel Rate [g.s-1] = MAF [g.s-1] * (Fuel-Air commanded equivalence ratio (Phi) / Stoichiometric air-to-fuel ratio)
    Stoichiometric air-to-fuel ratio for gasoline = 14.7:1
        
    Output:
        fc_out -- Fuel Rate [g.s-1]
    '''
    stoichiometric_air_to_fuel_ratio = 14.7
    fc_out = maf * phi / stoichiometric_air_to_fuel_ratio
    
    return fc_out

def emission_ppm_to_mg_per_m3(emission_name, emission_in, temperature, barometric_press):
    '''
    Convert emission [ppm] to emission [mg.m-3].
    
    emission_out [mg.m-3] = emission_in [ppm] * (molecular_weight_of_gas / 22.4) * (273 / (273 + temperature)) * (10 * barometric pressure / 1013)
    '''
    molecular_weight_of_gas = {'co2' : 44.01}
    emission_out = emission_in * (molecular_weight_of_gas[emission_name] / 22.4) * (273 / (273 + temperature)) * (10 * barometric_press / 1013)
    
    return emission_out

def emission_rate(emission_in, maf):
    '''
    Convert emission [mg.m-3] to emission rate [kg.s-1].
    
    emission_out [kg.s-1] = emission_in [mg.m-3] * 10^(-9) * MAF [g.s-1] / Air density
    Air density = 1.2929 [kg.m-3]
    '''
    air_density = 1.2929
    emission_out = emission_in * 10**(-9) * maf / air_density
    
    return emission_out

def generate_new_variables(df_in, gasoline_density = 718):
    '''
    Generate new variables.
    '''
    df = df_in.copy()

    df['Fuel Rate1 [g.s-1]'] = fuel_rate1(df['Fuel Rate [l.h-1]'], gasoline_density)
    df['Fuel Rate2 [g.s-1]'] = fuel_rate2(df['Mass Air Flow Rate [g.s-1]'], df['Phi'])
  
    if 'CO2 Conc. [%]' in df.columns:
        df['CO2 Conc. [ppm]'] = df['CO2 Conc. [%]'] * 10**4
        df['CO2 Conc. [mg.m-3]'] = emission_ppm_to_mg_per_m3('co2', df['CO2 Conc. [ppm]'], df['Temperature [c]'], df['Barometric Pressure [kpa]'])
        df['CO2 Rate [kg.s-1]'] = emission_rate(df['CO2 Conc. [mg.m-3]'], df['Mass Air Flow Rate [g.s-1]'])
        df['CO2 Rate [g.s-1]'] = df['CO2 Rate [kg.s-1]'] * 10**3

    return df

