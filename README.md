# BlueSkyAbove_Hackathon
Submission for the Blue Sky Above Challenge conducted by IEEE. <br />
https://www.hackerearth.com/challenges/hackathon/ieee-machine-learning-hackathon/

Project team: 
>> Ditipriya Gorai <br />
>> Chahbaz Aman

Running Procedure:
>> 1. Clone the repository.<br />
>>    This is necessary to download/save large satellite data files in the data/L2 folder.
>> 2. Run the [BlueSkyAbove_Solution.ipynb](https://github.com/Chahbaz-Aman/BlueSkyAbove_Hackathon/blob/main/BlueSkyAbove_Solution.ipynb) notebook.

Solution Overview:
>> The model implements a Random Forest Regressor as the NO2 estimator. <br />
>> 7 days prior satellite data is stored in data/L2. For a specified time and place for NO2 estimation, the most recent reliable satellite data is searched <br />
>> and extracted from the data files. This 2D spatial data is compressed into a 1D array of float values fed to the regressor.
>> Further, map data from OpenStreetMap for the place is downloaded and features are extracted from the image in terms of land-use scores.
>> The regressor uses the satellite data, land-use scores and time data to estimate NO2 concentration in the air. 
>> 
>> <img src="https://user-images.githubusercontent.com/87090353/151754048-13408859-2d64-416f-830a-aaf94a87b762.png" style="float:center" height="500" width="400"> <br />
>> Figure: Solution Overview

