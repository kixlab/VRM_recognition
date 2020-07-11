# VRM_recognition (this repo is for sharing with other labs) 

##### 4th year's product: korean VRM classifier 

##### All dependencies can be found in ```environment.yml```. Thus, first activate your own virtual environment and install the dependencies using ```$ conda env create -f environment.yml```.
##### This classifier takes korean sentences as input and outputs the corresponding VRM labels through (1) kor-eng translation module and (2) eng VRM classifier. To run the translation module, visit https://www.ncloud.com/, obtain your own NAVER PAPAGO ID and KEY, and enter the ID and KEY in the following:

```
# in test.py 
client_id = "" ## enter your id 
client_secret = "" ## enter your key
```
##### Make sure that ```word2vec_from_glove.bin``` in the folder ```data```.
##### To test the classifier on ```relabeling.csv```, run ```$ python test.py ```. The test result can be found in ```output.csv```.

##### * Caveat: the randomness may come from the translation module when testing. 
##### * Update: Form accuracy: 71.7% & Intent accuracy: 76.7%
