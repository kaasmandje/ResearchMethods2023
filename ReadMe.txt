CSV-files
The .csv files represent the retrieved data from the Husky robot.
The letter stands for the route and the number for the repetition.
In Dataformat_measurements.csv the original filenames can be seen.
This document can also be used to use for future analysis. 
By just editing the names, other files can directly be read out.

Python scripts
The DataProcessor.py file is the main Python script we worked in.
datascript.py is the script we received from last year’s group.

All the data DataProcessor.py produces gets exported to the folder 'Output'.
Because we worked in the cloud sometimes errors occurred with file paths in Python.
To overcome this error the file path was manually defined. 
Therefore, one should define this for itself.

Original Data
The original unedited data we received from our supervisor (Koen van Boheemen MSc)
is placed in the folder 'Data meting 8-6-2023' which refers to the 8th of June 2023,
the date at which the experiment took place, and the data was collected. 
Our supervisor converted the .BAG files that ROS (Robot Operating System) creates to .csv files.

The reason that the date of the creation of the data files is not 2023-06-08 is because there was an error in route A6.
Therefore, our supervisor exported all the data again. 
Unfortunately, this didn't solve the problem and we are not sure on what causes A6 to not work.

For any inconvenience or questions, you may contact Evan Ackermans (evan.ackermans@wur.nl) 
