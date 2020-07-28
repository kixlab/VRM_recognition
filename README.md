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

##### Credits
This application uses VRM speech act dataset. You can find the information in [this publication](https://d1wqtxts1xzle7.cloudfront.net/33842897/describing_talk.pdf?1401616826=&response-content-disposition=inline%3B+filename%3DDescribing_talk_A_taxonomy_of_verbal_res.pdf&Expires=1595947924&Signature=BEANFlSEMIAJDRIV3~1UshSc1~e~D3senlHAbMn4y~SdUgYY-jvDpLCLz8GZdP6loY3rnoUFRZYbJEGL6ZYX2wTk1YlK5z4L-3JmfUMnEBk4qxnJKMtNuTOCpCPzxHrnvacRKAaIkNndW8SdnPnzzBgyJFV16EgdBMi0YnthBjJZOmFC2t3aG3L8WiJFwAXXsr5rgj7vH52p67HKmSRHG9fFgMU~riHU2wm4r24Ss4YHuT0mLlp3M27zaKEpo2Lnw5fzo2Z5Z3S3h5asXCB9efCISD-ZQLR1cnaBGqCdQC757Cjk7AHF8TNR3pXiQionQbQfgMA6o01ABdj6S6hDSQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA). We acknowledge and are grateful to professor William B. Stiles for releasing dataset.
