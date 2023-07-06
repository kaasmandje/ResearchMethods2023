#!/bin/bash

#Runs script on each file in folder based on extension

#Built by Koen van Boheemen (WUR)
#Version: 2.0.2 (07-11-2018)

#dependencies:
#python2 (should be manually installed on Ubuntu 18)
#opencv (sudo apt-get install python-opencv)
#ros (on 16: http://wiki.ros.org/kinetic/Installation/Ubuntu On 18: http://wiki.ros.org/melodic/Installation/Ubuntu)

#Usage on Ubuntu 16/18/20:
#paste exportFolder.sh and ExportRealsenseRosbag.py (or another script) to folder containing bagfile(s)
#cd terminal to folder containing bagfile(s)
#run script by typing: . export.sh
#script automatically creates folders per bagfile

#Known issues:
#None 

FILES=*
#iterate through files
for file in $FILES
do
	#Get filename, extension
	fullfilename=$(basename -- "$file")
	filename="${fullfilename%.*}"
	extension="${fullfilename##*.}"
	
	#check extension to only select .bag files
	if [[ $extension =~ ^(bag)$ ]]
	then
		echo "Processing $fullfilename"
		#actual processing
		python ExportRosbags.py $fullfilename
		
		#check if folder does not yet exist
		#if [ ! -d "$filename" ]
		#then
		#	mkdir $filename
		#	
		#else
		#	echo "Skipping $fullfilename as folder already exists"
		#fi
	else
		echo "Skipping $fullfilename as it is not a .bag file"
	fi
done

