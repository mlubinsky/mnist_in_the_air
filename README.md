# mnist_in_the_air
The raw data was originally loaded from .txt file and saved in pickle files. Use mnist_air_models.ipynb to load the data from pickle files and train models.

* In total, 1397 gestures with accelerometer data (x, y, z) sampling at 50 Hz have been collected on Pixel and LG Android phones by 11/27/19.
* It requires downsampling (picking the middle part or picking randomly) and upsampling (padding zeros to the beginning). To be consistent with deployment with javascript, picking the middle part is used. 
* For the AIoT demo, a 3-layer neural network was used with the test accuracy stablizing at 75%. 
* In utils_functions.py, preprocess_v2 has "norm". This was not used for the demo. But based on analysis, SVM performs better when norm=True, equivalent to neural networks on this relatively small dataset. 
* The learning curve of SVM shows that adding more data could increase the accuracy. 
* All the raw data (json files) are saved in dropbox and downloaded to my laptop. The files were parsed and the data were saved in pickle files. 
