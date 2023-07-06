#!/usr/bin/env python

###################################################
#Script for exporting Husky Orchard bagfiles
###################################################

###################################################
#Meta
###
#
#
#Author: Koen van Boheemen
#Date: 04-06-2021
#Institute: WUR
#
#Version: 2.1
#Required:
#-Pyton3
#-Pyproj (apt-get install python3-pyproj)
#(possible)-nmea_msgs (apt-get install ros-[version]-nmea-msgs)
####################################################

####################################################
#Usage
###
#Ubuntu 16.04 / 18.04 / 20.04 with ROS installed
#cd to folder containing bagfile
#execute: python3 ExportRosbagsThesis.py [bagfilename]
#
#current config: to export data recorded to 1 bagfile for GPS, IMU & Odometry

####################################################
#Imports
###
import sys
import rosbag
from datetime import datetime as dt
import numpy as np
import os
import math
import pyproj

#####################################################

#####################################################
#Settings
###


#####################################################

#####################################################
#Main script
###

#Open bagfile
bagFileName = sys.argv[1]
bag = rosbag.Bag(bagFileName)



#filename variable for creating folders + files
fileName = bagFileName[:-4]

#Prepare logfile
#logLine = "Timestamp;GPSLatitude;GPSLongitude;GPSQuality\r\n"
log = open(fileName+".csv", "w+")
#log.write(logLine)
#log.flush()

#Prepare reprojection to RD_New (EPSG:28992)
wgs84 = pyproj.Proj(init="EPSG:4326")
rdnew = pyproj.Proj(init="EPSG:28992")

#Loop for reading messages in bag one at a time
for topic, msg, t in bag.read_messages(topics=['husky_velocity_controller/cmd_vel', 'imu/data', 'nmea_sentence', 'odometry/filtered']):
#Select per topic (= per type)
	#GPS
	if topic == 'nmea_sentence':
		gpsFields = [field.strip(',') for field in msg.sentence.split(',')]
		#GGA sentence contains location information
		if gpsFields[0] == "$GPGGA":
			quality = int(gpsFields[6])
			
			
			#Convert time
			timeFloat = msg.header.stamp.secs + (msg.header.stamp.nsecs / 1000000000.0)
			GPSTime = dt.utcfromtimestamp(timeFloat).strftime('%Y:%m:%d %H:%M:%S.%f')
			
			#Convert coordinates
			latitude = float(gpsFields[2][0:2]) + (float(gpsFields[2][2:]) / 60.0)
			longitude = float(gpsFields[4][0:3]) + (float(gpsFields[4][3:]) / 60.0)
			
			rdlat, rdlon = pyproj.transform(wgs84, rdnew, longitude, latitude)

			
			#write logfile
			logLine = "GPS;" + GPSTime + ";" + repr(latitude) + ";" + repr(longitude) + ";" + repr(quality) + ";" + repr(rdlat) + ";" + repr(rdlon) + '\r\n'
			log.write(logLine)
			log.flush()
	
	
	elif topic == 'imu/data':
		#Convert time
		timeFloat = msg.header.stamp.secs + (msg.header.stamp.nsecs / 1000000000.0)
		IMUTime = dt.utcfromtimestamp(timeFloat).strftime('%Y:%m:%d %H:%M:%S.%f')

		#Orientation
		x = msg.orientation.x
		y = msg.orientation.y
		z = msg.orientation.z
		w = msg.orientation.w

		ysqr = y * y
	
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + ysqr)
		roll = math.degrees(math.atan2(t0, t1)) #rotation around x-axis of UM6
	
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.degrees(math.asin(t2)) #rotation around y-axis of UM6
	
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (ysqr + z * z)
		yaw = math.degrees(math.atan2(t3, t4)) #rotation around z-axis of UM6

		#Angular Velocity
		deltax = msg.angular_velocity.x
		deltay = msg.angular_velocity.y
		deltaz = msg.angular_velocity.z
		
		#Linear acceleration
		delta2x = msg.linear_acceleration.x
		delta2y = msg.linear_acceleration.y
		delta2z = msg.linear_acceleration.z
	
		logLine = "IMU;" + IMUTime + ";Orientation;" + repr(roll) + ";" + repr(pitch) + ";" + repr(yaw) + ";AngularVelocity;" + repr(deltax) + ";" + repr(deltay) + ";" + repr(deltaz) + ";LinearAcceleration;" + repr(delta2x) + ";" + repr(delta2y) + ";" + repr(delta2z) + "\r\n"
		log.write(logLine)
		log.flush()

	elif topic == 'odometry/filtered':
		timeFloat = msg.header.stamp.secs + (msg.header.stamp.nsecs / 1000000000.0)
		OdoTime = dt.utcfromtimestamp(timeFloat).strftime('%Y:%m:%d %H:%M:%S.%f')

		#position
		posx = msg.pose.pose.position.x
		posy = msg.pose.pose.position.y
		posz = msg.pose.pose.position.z

		deltaposx = msg.twist.twist.linear.x
		deltaposy = msg.twist.twist.linear.y
		deltaposz = msg.twist.twist.linear.z

		#Orientation
		x = msg.pose.pose.orientation.x
		y = msg.pose.pose.orientation.y
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w
		
		ysqr = y * y
	
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + ysqr)
		roll = math.degrees(math.atan2(t0, t1))
	
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.degrees(math.asin(t2))
	
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (ysqr + z * z)
		yaw = math.degrees(math.atan2(t3, t4))

		deltax = msg.twist.twist.angular.x
		deltay = msg.twist.twist.angular.y
		deltaz = msg.twist.twist.angular.z

		logLine = "Odometry;" + OdoTime + ";Position;" + repr(posx) + ";" + repr(posy) + ";" + repr(posz) + ";DeltaPosition;" + repr(deltaposx) + ";" + repr(deltaposy) + ";" + repr(deltaposz) +";Orientation;" + repr(roll) + ";" + repr(pitch) + ";" + repr(yaw) + ";AngularVelocity;" + repr(deltax) + ";" + repr(deltay) + ";" + repr(deltaz) + "\r\n"
		log.write(logLine)
		log.flush()
	else:
		pass

bag.close()
log.close()

print("Finished processing " + repr(bagFileName)[1:-1])








