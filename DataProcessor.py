###################################################

###################################################
#
#Date: 12-06-2023
#Institute: WUR
#Continuation of datascript.py
#Continuation from last year Research Methods for Biosystems Engineering
#Recieved from Jeroen Cox
#Version:
#
#
#Main improvements:
#Wout Hartveld and Evan Ackermans
#
#
#Currently working on:
#-Calibration for relative distances
#-Uniform data processing
#-Time bookholding
#-Movement detection
#-Land Marks
#-Also I would like to convert some code into functions, but this has low priority
#-Implementation of Classes 
#-Solving current demarcation in length calculation
####################################################

import pandas as pd
import math as m
import numpy as np
from collections import defaultdict
from scipy import integrate
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os
import re

#Global variables Initialiations
working_dir = os.getcwd()
"""
Currentpath is used because there is some uncertainty in current working directory. 
Everyone needs to make sure for themselves to look at this, as it is currently not consistent. 
We look into this

Currently IMU development of the code is out of consideration
"""
currentpath = 'C:/Users/evana/OneDrive - Wageningen University & Research/D1/Data meting 8-6-2023 Beunmap/'
output_folder = '/Output/'

if working_dir != currentpath:
    print("Please make sure 'currentpath' is defined correctly.\nYour current working directory is not equal to currentpath.\nIgnore this if you know what you are doing.")
dataformat_file = "Dataformat_measurements.csv"
# filename_input = input("Press enter to exit.\nWhich route you want to see? ")
# filename = currentpath+filename_input.upper()+'.csv'
# route = filename_input.upper()
# print("Processing route "+filename_input.upper()+" ...")

def route_retriever(currentpath,file): # Extract the filenames as a list
    df_files = pd.read_csv(currentpath+file,sep = ';')
    filenames = (df_files['Group']+df_files['Repetition'].astype(str)).tolist()
    return filenames

filenames = route_retriever(currentpath, dataformat_file)
del filenames[5] #A6 keeps giving a error, we don't know why

df_cor_fact = pd.DataFrame(columns=['Route','Cor_ODO','Cor_IMU'])
df_total_results = pd.DataFrame(columns=['Route', 'Repetition', 'Time', 'Deviation', 'Derivative', 'Length_ODO', 'Length_GPS'])
writer = pd.ExcelWriter(currentpath+output_folder+'Combined_results.xlsx', engine='xlsxwriter')

for i in range(len(filenames)):
    filename_input = filenames[i]
    filename = currentpath+filename_input.upper()+'.csv'
    route = filename_input.upper()
    route_split = re.findall(r'[A-Za-z]+|\d+', route)
    # writer = pd.ExcelWriter(currentpath+output_folder+route+'_errors.xlsx', engine='xlsxwriter')
    print("Processing route "+filename_input.upper()+" ...")
    df = pd.read_csv(filename,sep = ';')
    df.head(4)
    
    def parse_line(split):
        '''
        Parser, not relevant to the math
        '''
        r = {}
        r[split[0]] = [float(o) for o in split[1:4]]
        r[split[4]] = [float(o) for o in split[5:8]]
        r[split[8]] = [float(o) for o in split[9:12]]
        if len(split) > 13:
            r[split[12]] = [float(o) for o in  split[13:]]
        return r
    
    
    def new_position_matrix(x, a, delta_t, angle):
        '''
        Matrices from DOI: 10.1109/IPIN.2013.6817887
        :x is a numpy array with [pos_x, pos_y, v_x, v_y]
        but when returning it returns [unknown_x, unknown_y, pos_x, pos_y]
        so the function is treated as a black box
    
        a = numpy array with [ax, ay]
        '''
        R = np.matrix([[m.cos(angle), -m.sin(angle)],
                       [m.sin(angle), m.cos(angle)]])
    
        PHI = np.matrix([[1, 0, delta_t, 0],
                         [0, 1, 0, delta_t],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    
        B = np.matrix([[0.5 * (delta_t ** 2), 0],
                       [0, 0.5 * (delta_t ** 2)],
                       [delta_t, 0],
                       [0, delta_t]])
        Ra = np.dot(R, a)
        PHIx = PHI.dot(x)
        # print('xdot: ', xPHI)
        BR = B.dot(R)
        BRa = BR.dot(a)
        new_x = PHIx + BRa
        return new_x
    #still parsing, not relevant to the math
    odometry,gps,imu = {},{},{}
    
    with open(filename) as  f:
        first_time = True
        for line in f:
            splitted = line.replace('\n','').split(';')
            ts = dt.strptime(splitted[1],'%Y:%m:%d %H:%M:%S.%f')
            if first_time == True:
                starttime = ts #.timestamp()
                first_time = False
            elif splitted[0] == "Odometry":
                odometry[ts] = parse_line(splitted[2:])
            elif splitted[0] == "GPS":
                gps[ts] =[float(o) for o in splitted[-2:]]
                pass
            elif splitted[0] == "IMU":
                imu[ts] = parse_line(splitted[2:])
    imu_df = pd.DataFrame(imu).T
    
    
    #getting odometry positioning
    df_odo = pd.DataFrame(odometry).T
    df_odo_pos = pd.DataFrame({t : o['Position'] for t,o in odometry.items()}).T
    df_odo_pos = df_odo_pos.rename(columns = {0:'x',1:'y',2:"z",3:'length'})
    df_odo_pos['x'] = df_odo_pos['x'] - df_odo_pos['x'][0]
    df_odo_pos['y'] = df_odo_pos['y'] - df_odo_pos['y'][0]
    length = np.sqrt((df_odo_pos['x'].diff()**2) + (df_odo_pos['y'].diff()**2)).cumsum()
    df_odo_pos['length'] = np.sqrt((df_odo_pos['x'].diff()**2) + (df_odo_pos['y'].diff()**2)).cumsum()
    
    # Because the first value otherwise will be NaN, this makes sure other functions run smootly
    df_odo_pos['length'] = df_odo_pos['length'].fillna(0) 
    df_odo_pos['time_passed'] = df_odo_pos.index
    df_odo_pos['time_passed'] = pd.to_datetime(df_odo_pos['time_passed'], format='%Y-%m-%d %H:%M:%S.%f')
    df_odo_pos['time_passed'] = (df_odo_pos['time_passed'] - starttime).dt.total_seconds()
    
    df_odo_pos = df_odo_pos.reset_index()
    
    
    #Building gps datagrame
    df_gps = pd.DataFrame(gps).T
    df_gps = df_gps.reset_index()
    df_gps[0] = df_gps[0] -df_gps[0][0]
    df_gps[1] = df_gps[1] -df_gps[1][0]
    df_gps = df_gps.rename(columns = {0: 'x', 1: 'y'})
    df_gps['dx'] = abs(df_gps['x'] - df_gps[ 'x'].shift(1))
    df_gps['dy'] = abs(df_gps['y'] - df_gps[ 'y'].shift(1))
    #df_gps['length'] = np.sqrt(df_gps['x']**2 + df_gps['y']**2)
    
    df_gps['length'] = (df_gps['dx']**2 + df_gps['dy']**2)**0.5
    lst = []
    for i in range(len(df_gps['length'])):
        if i == 0:
            lst.append(0)
        else:
            temp = df_gps['length'][i]+lst[i-1]
            lst.append(temp)
    df_gps['length'] = lst
    df_gps['time_passed'] = pd.to_datetime(df_gps['index'], format='%Y-%m-%d %H:%M:%S.%f')
    df_gps['time_passed'] = (df_gps['time_passed'] - starttime).dt.total_seconds()
    
    #building orientation dataframe                 #Why do we call it orientation?
    df = pd.DataFrame({t : o['Orientation'] for t,o in imu.items()}).T
    df = df.rename(columns = {0:'x',1:'y',2:"z"})
    df = df.reset_index()
    df['x'] = df['x']- df['x'][0]
    df_orient = df
    df_orient['z'] = df_orient['z'] - df_orient['z'][0]
    df_orient = df.select_dtypes(exclude=['datetime']).multiply(2*m.pi/(360)) #word gedaan omdat anders multiplicatie niet werkt, maar nu mist er data
    
    #This is where the utilisation of new_position_matrix() is started
    x = np.zeros(4)
    i = 0
    lst_array = []
    lst_time = []
    current_time = None # I don't think this is exactly correct for time bookholding (Evan) but for this function it works
    time_past = 0
    for time, info in imu.items():  #this loop is mostly for converting the indextime to the
        if current_time is None:
            current_time = time
            continue
        delta_t = (time - current_time).total_seconds()     #getting delta t in seconds
        current_time = time
        time_past += delta_t
        a = np.array([imu_df['LinearAcceleration'][i][1], imu_df['LinearAcceleration'][i][2]])  #builds the acceleration array
        #a = np.array([0, 0])
        phi = df_orient['z'][i]     #getting the angle from the dataset
        #print(phi)
        if i == 0:                  #this is purely to fix a bug
            delta_t = 0
        x = new_position_matrix(x, a, delta_t, phi)
        x = np.asarray(x).flatten()     #converting from matrix to array
        array = x
        lst_array.append(array)             #from here on this is for building the dataset, not relevant to the math
        finalarray = np.array(lst_array)
        i += 1
        lst_time.append(time_past)
    
    #we miss some data from df to df_pos, and we have some unused dataframes in our variable list.
    #I think it could be done more efficient (Evan)
    df_pos = pd.DataFrame(finalarray, columns = ['tempx', 'tempy', 'x', 'y']) #'length','time_passed'
    df_pos['x'] = df_pos['x']#*-1 #removed this (Evan)
    #df_pos['length'] = m.sqrt(df_pos['x']**2 + df_pos['y']**2)
    df_pos['length'] = df_pos.apply(lambda row: m.sqrt(row['x']**2 + row['y']**2), axis=1)
    df_pos['time_passed'] = lst_time # I don't really trust this time 
    
    def calculate_initial_orientation(data, delta_len=15, min_len=0.5): 
        count = 0
        while count < len(data['length']):
            if data['length'][count] < min_len: 
                count += 1
            else:
                break
        
        if count + delta_len < len(data):
            x1 = data['x'][count]
            y1 = data['y'][count]
            x2 = data['x'][count+delta_len] 
            y2 = data['y'][count+delta_len]
        else:
            raise ValueError("Not enough data points for angle calculation.")
        
        angle = m.atan2(y2 - y1, x2 - x1) 
        return angle
    
    def distance_calibration(sensor_df, gps_df, d_len_gps=15, sensor_freq=2, min_len=0.5): 
      count_gps = 0
      while count_gps < len(gps_df['length']):
        if gps_df['length'][count_gps] < min_len: 
                 #0.05 is an arbritrary value to detect that the husky starts to move
                 #Maybe implement a percentage of the total driven length
                 count_gps += 1
        else:
            break
         
        if count_gps + d_len_gps < len(gps_df): 
             x1 = gps_df['x'][count_gps]
             y1 = gps_df['y'][count_gps]
             x2 = gps_df['x'][count_gps+d_len_gps] 
             y2 = gps_df['y'][count_gps+d_len_gps]
        else:
            raise ValueError("Not enough GPS points for the distance-calibration.")
        
        gps_dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
        start_time_gps = gps_df['time_passed'][count_gps]
        time_passed_gps = gps_df['time_passed'][count_gps+d_len_gps]-gps_df['time_passed'][count_gps]
        avg_dist_gps = gps_dist/time_passed_gps
        
        count_sensor = 0
        while sensor_df['time_passed'][count_sensor] < start_time_gps: #ook niet echt heel waterdicht dit
            count_sensor += 1
         
        if count_sensor + d_len_gps*sensor_freq < len(sensor_df): 
             x1 = sensor_df['x'][count_sensor]
             y1 = sensor_df['y'][count_sensor]
             x2 = sensor_df['x'][count_sensor+d_len_gps*sensor_freq] # To correct for sampling frequency
             y2 = sensor_df['y'][count_sensor+d_len_gps*sensor_freq]
        else:
            raise ValueError("Not enough sensor points for the distance-calibration.")
        
        sensor_dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
        time_passed_sensor = sensor_df['time_passed'][count_sensor+d_len_gps*sensor_freq]-sensor_df['time_passed'][count_sensor]
        avg_dist_sensor = sensor_dist/time_passed_sensor
        cor_factor = avg_dist_gps/avg_dist_sensor   
        return cor_factor
    
    def rotate_matrix(angle, df_sensor):
        x_list = df_sensor['x'].values.astype(float)
        y_list = df_sensor['y'].values.astype(float)
        timestamps = df_sensor['time_passed'].values  # Extract timestamps
        lengths = df_sensor['length'].values
        
        df_rotated_odo = pd.DataFrame(columns=['time_passed', 'x', 'y', 'length'])  # Include 'time_passed' column
        
        for i in range(x_list.shape[0]):
            x = x_list[i]
            y = y_list[i]
            timestamp = timestamps[i]       # Get the corresponding timestamp
            length = lengths[i]
            
            x_new = x * m.cos(angle) - y * m.sin(angle)
            y_new = y * m.cos(angle) + x * m.sin(angle)
            
            new_line = pd.DataFrame({"time_passed": [timestamp], "x": [x_new], "y": [y_new], "length": [length]}, index=[i])
            df_rotated_odo = pd.concat((df_rotated_odo, new_line), axis=0)
        
        return df_rotated_odo
    
    def distance_correction(cor_fact, df_sensor):
        x_list = df_sensor['x'].values.astype(float)
        y_list = df_sensor['y'].values.astype(float)
        df_sensor_dist_cor = pd.DataFrame(columns = ['x','y','length'])
        
        for i in range(x_list.shape[0]):
            x = x_list[i]
            y = y_list[i]
            x_new   = x * cor_fact
            y_new   = y * cor_fact
            length  = np.sqrt((x_new**2)+(y_new**2))
            new_line = pd.DataFrame({"x":[x_new], "y":[y_new], "length":[length]})
            df_sensor_dist_cor = pd.concat((df_sensor_dist_cor, new_line), axis=0)
        return df_sensor_dist_cor
    
    angle_gps = calculate_initial_orientation(df_gps,       delta_len = 15)
    angle_pos = calculate_initial_orientation(df_pos,       delta_len = 30)
    angle_odo = calculate_initial_orientation(df_odo_pos,   delta_len = 60)
    
    #Works nice
    df_rotated_odo = rotate_matrix(angle_gps-angle_odo,df_odo_pos)
    
    #Doesn't work nice
    #df_rotated_pos = rotate_matrix(angle_pos-angle_gps-m.radians(75),df_pos)
    #cor_fact_odo = distance_calibration(df_odo_pos, df_gps, sensor_freq = 4)
    #cor_fact_imu = distance_calibration(df_pos, df_gps, sensor_freq = 2)
    #df_cor_fact = df_cor_fact.append({'Route': route, 'Cor_ODO': cor_fact_odo,'Cor_IMU': cor_fact_imu}, ignore_index=True)
    
    #df_corrected_pos = distance_correction(cor_fact_imu, df_rotated_pos)
    
    def calc_deviation_derivative(df_gps, df_sensor, route=route, excel_writer=None):
        # Interpolate df_sensor coordinates
        interpolated_x      = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['x'])
        interpolated_y      = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['y'])
        interpolated_length = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['length'])
        
        # Calculate deviation (Euclidean distance)
        deviation_x = interpolated_x - df_gps['x']
        deviation_y = interpolated_y - df_gps['y']
        deviation = np.sqrt(deviation_x**2 + deviation_y**2)
    
        # Calculate the derivative (slope) using finite differences
        time_diff = np.diff(df_gps['time_passed'])
        deviation_diff = np.diff(deviation)
        derivative = deviation_diff / time_diff
    
        # Plot deviation
        fig_deviations, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=250, sharex=True)
        title = ('Route: %s Errors Odometry')% route
        
        # Plot deviation
        ax1.plot(df_gps['time_passed'], deviation, label='Deviation [m]')
        ax1.set_ylabel('Deviation (Euclidean distance)')
        ax1.set_title('Deviation as Euclidean distance over time')
        ax1.legend()
        ax1.grid(True)
        
        dark_orange = (0.9, 0.4, 0)
        # Plot derivative
        ax2.plot(df_gps['time_passed'][:-1], derivative, label='Derivative [m]', color=dark_orange)
        ax2.set_title('Error slope (derivative) over time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Derivative')
        ax2.legend()
        ax2.grid(True)
        
        fig_deviations.suptitle(title, y=0.96, fontsize=18)   
        plt.savefig(currentpath+output_folder+route+'_errors.png', dpi=250, bbox_inches='tight')         
        plt.show()
        # Create a DataFrame to store the deviation and derivative values
        df_results = pd.DataFrame({
            'Time'          : df_gps['time_passed'],
            'Deviation'     : deviation,
            'Derivative'    : np.concatenate(([np.nan], derivative)),
            'Length_ODO'    : interpolated_length,
            'Length_GPS'    : df_gps['length']
        })
        
        df_total_results = pd.DataFrame({
            'Time': df_gps['time_passed'],
            'Deviation': deviation,
            'Derivative': np.concatenate(([np.nan], derivative)),
            'Length_ODO': interpolated_length,
            'Length_GPS': df_gps['length'],
            'Route': route_split[0],
            'Repetition': route_split[1] 
        })
    
        # Write the DataFrame to the Excel file
        if excel_writer is not None:
            df_results.to_excel(excel_writer, sheet_name=route, index=False)
        return df_total_results
        

    fig, ax = plt.subplots(figsize=(10, 7), dpi=250)
    
    # Plot GPS, Odometry, and IMU on the subplot
    ax.plot(df_gps['x'], df_gps['y'], label='RTK-GPS')
    ax.plot(df_rotated_odo['x'], df_rotated_odo['y'], label='Odometry')
    #ax.plot(df_corrected_pos['x'], df_corrected_pos['y'], label='IMU') 
    # We exclude it for now as the results are not reliable
    
    # Set the title and legend
    title = 'Route: %s' % route
    ax.set_title(title, fontsize=18)
    ax.legend()
    plt.savefig(currentpath+output_folder+route+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    
    # Run the plot function
    route_result = calc_deviation_derivative(df_gps, df_rotated_odo, excel_writer = writer)
    # Get the last row of the DataFrame
    last_row = route_result.tail(1).copy()
    # Add the 'Route' and 'Repetition' columns to the last row
    last_row['Route'] = route_split[0]
    last_row['Repetition'] = route_split[1]
    # Concatenate the last row with the df_total_results DataFrame
    df_total_results = pd.concat([df_total_results, last_row], ignore_index=True)
    

# Save the df_total_results DataFrame to an Excel file
df_total_results.to_excel(writer, sheet_name="All Routes", index=False)
# Close the Excel file
writer.save() 


print("Done.")
    
    
"""

    # Although it is inefficient coding this way, for now it works and time is ticking (Evan)
    def total_results(df_gps, df_sensor, excel_writer=None, route=route, route_split=route_split):
        # Interpolate df_sensor coordinates
        interpolated_x = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['x'])
        interpolated_y = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['y'])
        interpolated_length = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['length'])
    
        # Calculate deviation (Euclidean distance)
        deviation_x = interpolated_x - df_gps['x']
        deviation_y = interpolated_y - df_gps['y']
        deviation = np.sqrt(deviation_x ** 2 + deviation_y ** 2)
    
        # Calculate the derivative (slope) using finite differences
        time_diff = np.diff(df_gps['time_passed'])
        deviation_diff = np.diff(deviation)
        derivative = deviation_diff / time_diff
        
        # Create a DataFrame to store the deviation and derivative values
        df_results = pd.DataFrame({
            'Time': df_gps['time_passed'],
            'Deviation': deviation,
            'Derivative': np.concatenate(([np.nan], derivative)),
            'Length_ODO': interpolated_length,
            'Length_GPS': df_gps['length'],
            'Route': route_split[0],
            'Repetition': route_split[1] 
        })
    
        # Write the DataFrame to the Excel file
        if excel_writer is not None:
            df_results.to_excel(excel_writer, sheet_name='Results', index=False)
    
        return df_results
    
    df_results = total_results(df_gps, df_rotated_odo, route=route, excel_writer=writer)
    # Get the last row of the DataFrame
    last_row = df_results.tail(1).copy()
    # Add the 'Route' and 'Repetition' columns to the last row
    last_row['Route'] = route_split[0]
    last_row['Repetition'] = route_split[1]








    #plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize= (10,7), dpi=250)
    df_pos.plot( 'x', 'y', kind= 'line', ax = ax1)      #plot IMU
    ax1.set_title('IMU')
    
    #df_pos.plot('vx', 'vy', kind = 'line')
    df_gps.plot('x','y', ax = ax2, kind = 'line')       #plot GPS
    ax2.set_title('RTK-GPS')
    
    df_odo_pos['y'] = df_odo_pos['y']* -1
    df_odo_pos.plot('x', 'y', ax= ax3, kind = 'line')   #plot ODO
    ax3.set_title('Odometry')
    
    df_gps.plot('x', 'y', ax=ax4, kind='line', label='RTK-GPS') #plot overlaps
    df_rotated_odo.plot('x', 'y', ax=ax4, kind='line', label='Odometry')
    df_corrected_pos.plot('x','y', ax=ax4, kind='line', label='IMU')
    ax4.set_title('GPS, Odometry and IMU')
    ax4.legend()
    title = 'Route: %s' %route
    fig.suptitle(title, y=0.96, fontsize=18)
    #result = filename_input.upper()+".png"
    #plt.savefig(result)
    #print("Image '"+result+"' saved to current directory")
    plt.show()

Was makkelijker geweest als we de metingen eerst calibreerde (eerste 5 seconden ofzo) 
en pas daarna verder gingen met uitlezen, zodat je de rest van de metingen maar één keer
hoeft te processen. Maar binnen de beperkte tijd die we hadden was dit niet mogelijk en zijn
we verder gegaan met de code die we hadden gekregen.


def plot_tot_function(df_gps, df_sensor, rotated_odo_data, corrected_pos_data, gps_data=df_gps, route=route):
    # Create a figure with a grid layout of 2 rows and 2 columns
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2)

    # Interpolate df_sensor coordinates
    interpolated_x = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['x'])
    interpolated_y = np.interp(df_gps['time_passed'], df_sensor['time_passed'], df_sensor['y'])

    # Calculate deviation (Euclidean distance)
    deviation_x = interpolated_x - df_gps['x']
    deviation_y = interpolated_y - df_gps['y']
    deviation = np.sqrt(deviation_x**2 + deviation_y**2)

    # Calculate the derivative (slope) using finite differences
    time_diff = np.diff(df_gps['time_passed'])
    deviation_diff = np.diff(deviation)
    derivative = deviation_diff / time_diff

    # Plot deviation
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(deviation['time_passed'], deviation['deviation'], label='Deviation')
    ax1.set_ylabel('Deviation (Euclidean distance)')
    ax1.set_title('Deviation as Euclidean distance over time')
    ax1.legend()
    ax1.grid(True)

    dark_orange = (0.9, 0.4, 0)
    # Plot derivative
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(derivative['time_passed'][:-1], derivative['derivative'], label='Derivative', color=dark_orange)
    ax2.set_title('Error slope (derivative) over time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Derivative')
    ax2.legend()
    ax2.grid(True)

    # Plot GPS, Odometry, and IMU on the left
    ax4 = fig.add_subplot(grid[:, 1])
    gps_data.plot('x', 'y', ax=ax4, kind='line', label='RTK-GPS')
    rotated_odo_data.plot('x', 'y', ax=ax4, kind='line', label='Odometry')
    corrected_pos_data.plot('x', 'y', ax=ax4, kind='line', label='IMU')
    ax4.set_title('GPS, Odometry, and IMU')
    ax4.legend()

    # Add a title to the figure
    title = 'Route: %s' % route
    fig.suptitle(title, y=0.96, fontsize=18)

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
    
plot_tot_function(df_gps, df_rotated_odo, df_gps, df_rotated_odo, df_corrected_pos)

"""