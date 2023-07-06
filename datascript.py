###################################################

###################################################
#
#Date: 12-06-2023
#Institute: WUR
#
#Version:
#
####################################################

import pandas as pd
import math as m
import numpy as np
from collections import defaultdict
from scipy import integrate
import matplotlib.pyplot as plt
import os

while True:
    filename_input = input("Press enter to exit.\nWhich route you want to see? ")
    # Perform actions with the user input
    filename = filename_input.upper()+'.csv'
    print("Processing route "+filename_input.upper()+" ...")
    route = "Route: " + filename.split('.')[0]        
    
    # Exit condition
    if filename_input.lower() == 'exit' or filename_input.lower() == '':
        print("Exiting...")
        break
    #filename = input("Which route you want to see?").upper()+'.csv'
    #route = "Route: " + filename.split('.')[0]
    
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
    
    from datetime import datetime as dt
    with open(filename) as  f:
        for line in f:
            splitted = line.replace('\n','').split(';')
            ts = dt.strptime(splitted[1],'%Y:%m:%d %H:%M:%S.%f')
            if splitted[0] == "Odometry":
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
    df_odo_pos= df_odo_pos.rename(columns = {0:'x',1:'y',2:"z"})
    df_odo_pos['x'] = df_odo_pos['x'] - df_odo_pos['x'][0]
    df_odo_pos['y'] = df_odo_pos['y'] - df_odo_pos['y'][0]
    df_odo_pos = df_odo_pos.reset_index()
    
    #Building gps datagrame
    df_gps = pd.DataFrame(gps).T
    df_gps = df_gps.reset_index()
    df_gps[0] = df_gps[0] -df_gps[0][0]
    df_gps[1] = df_gps[1] -df_gps[1][0]
    df_gps = df_gps.rename(columns = {0: 'x', 1: 'y'})
    df_gps['dx'] = abs(df_gps['x'] - df_gps[ 'x'].shift(1))
    df_gps['dy'] = abs(df_gps['y'] - df_gps[ 'y'].shift(1))
    df_gps['length'] = (df_gps['dx']**2 + df_gps['dy']**2)**0.5
    lst = []
    for i in range(len(df_gps['length'])):
        if i == 0:
            lst.append(0)
        else:
            temp = df_gps['length'][i]+lst[i-1]
            lst.append(temp)
    df_gps['length'] = lst
    
    #building orientation dataframe
    df = pd.DataFrame({t : o['Orientation'] for t,o in imu.items()}).T
    df= df.rename(columns = {0:'x',1:'y',2:"z"})
    df = df.reset_index()
    df['x'] = df['x']- df['x'][0]
    df_orient = df
    df_orient['z'] = df_orient['z'] - df_orient['z'][0]
    df_orient = df.select_dtypes(exclude=['datetime']).multiply(2*m.pi/(360))
    
    
    #This is where the utilisation of new_position_matrix() is started
    x = np.zeros(4)
    i = 0
    lst_array = []
    lst_time = []
    current_time = None
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
        if i == 0:          #this is purely to fix a bug
            delta_t = 0
        x = new_position_matrix(x, a, delta_t, phi)
        x = np.asarray(x).flatten()     #converting from matrix to array
        array = x
        lst_array.append(array)             #from here on this is for building the dataset, not relevant to the math
        finalarray = np.array(lst_array)
        i += 1
        lst_time.append(time_past)
    
    df_pos = pd.DataFrame(finalarray, columns = ['tempx', 'tempy', 'x', 'y'])
    df_pos['x'] = df_pos['x']*-1
    df_pos['time'] = lst_time
    
    
    #plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize= (10,6), dpi=250)
    df_pos.plot( 'x', 'y', kind= 'line', ax = ax1)      #plot IMU
    ax1.set_title('IMU')
    #df_pos.plot('vx', 'vy', kind = 'line')
    df_gps.plot('x','y', ax = ax2, kind = 'line')       #plot GPS
    ax2.set_title('RTK-GPS')
    df_odo_pos['y'] = df_odo_pos['y']* -1
    df_odo_pos.plot('x', 'y', ax= ax3, kind = 'line')   #plot ODO
    ax3.set_title('Odometry')
    fig.suptitle(route, y=0.97, fontsize=18)
    result = filename_input.upper()+".png"
    plt.savefig(result)
    print("Image '"+result+"' saved to current directory")
    plt.show()
    
    """""
    # Save the figure in a specific directory
    save_dir = '/Results'                       # Specify the directory path here
    filename = filename_input.upper() + '.png'  # Specify the desired filename
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure in the specified directory
    fig_path = os.path.join(save_dir, filename)
    plt.savefig(fig_path)
    
    # Show the saved file path
    print(f"Figure saved at: {fig_path}")
    
    plt.show()
    
    # Plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(15, 10))
    
    df_pos.plot('x', 'y', kind='line', ax=ax1)  # Plot IMU
    ax1.set_title('IMU')
    
    df_gps.plot('x', 'y', ax=ax2, kind='line')  # Plot GPS
    ax2.set_title('RTK-GPS')
    
    df_odo_pos.plot('x', 'y', ax=ax3, kind='line')  # Plot ODO
    ax3.set_title('Odometry')
    
    # Calculate difference between GPS and IMU
    df_gps_imu = df_gps.copy()
    df_gps_imu['x_diff'] = df_gps_imu['x'] - df_pos['x']
    df_gps_imu['y_diff'] = df_gps_imu['y'] - df_pos['y']
    
    # Plot GPS-IMU difference
    df_gps_imu.plot('x_diff', 'y_diff', ax=ax4, kind='line')
    ax4.set_title('GPS - IMU Difference')
    
    # Calculate difference between GPS and Odometry
    df_gps_odo = df_gps.copy()
    df_gps_odo['x_diff'] = df_gps_odo['x'] - df_odo_pos['x']
    df_gps_odo['y_diff'] = df_gps_odo['y'] - df_odo_pos['y']
    
    # Plot GPS-Odometry difference
    df_gps_odo.plot('x_diff', 'y_diff', ax=ax5, kind='line')
    ax5.set_title('GPS - Odometry Difference')
    
    fig.suptitle(route, y=1.01, fontsize=18)
    plt.tight_layout()  # Add this line to adjust subplot spacing
    plt.show()
    """""



