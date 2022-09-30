# multiscale-voting-mechanism
Pytorch implementation of [Multiscale voting mechanism for rice leaf disease recognition under natural field conditions](https://doi.org/10.1002/int.23081).  
This project aims to build a general framework for multiscale-voting-mechanism in rice leaf disease classification.  
## Requirements
* Python3.8  
* Pytorch 1.9.1  
* Cuda 11.1  
* Ubuntu20.04 Linux  
* CPU:Intel Xeon Silver 4210R processor×2  
* 32 GB memory×8  
* hard disk:RTX3090 24 GB graphics card×4  


## Usage
------
## Dataset preparation
The dataset used in this study were collected under different weather conditions, such as sunny, cloudy, and rainy. A total of 6046 rice leaf images were collected, including 1046 rice bacterial leaf blight images, 1053 rice blast images,1542 rice brown spot images, 823 rice sheath blight images, and 1582 healthy leaf images.The dataset is available at [https://pan.baidu.com/s/1lFDPeLjayBOCESdem_WQfA?pwd=7z3s.](https://pan.baidu.com/s/1lFDPeLjayBOCESdem_WQfA?pwd=7z3s)  
After downloading, the dataset should be placed at the "dataset" folder under the project, where the training and testing samples were defined in the responding text files.
For more details on our dataset, you can refer to our paper [Multiscale voting mechanism for rice leaf disease recognition under natural field conditions](https://doi.org/10.1002/int.23081). 

## Code Folder Description  
***- dataset:used to store data sets.***  
  
***- di.py: Directories (including image paths and real labels) used to generate training and test sets.***  
  
***- densenet：CNN model***    
-- densenet_train.py ：Run to get training and test results.  
-- densenet_results analysis.py: Confusion matrix, Precision, Recall and F1 score are obtained.  
-- class_indices_5.json：The category name that generates the confusion matrix.    
  
***- googlenet：CNN model***  
-- googlenet_train.py ：Run to get training and test results.  
-- googlenet_results analysis.py: Confusion matrix, Precision, Recall and F1 score are obtained.  
-- models-flops.py：Compute model's Params and FLOPs.  
  
***- resnet：CNN model***  
-- resnet_train.py：Run to get training and test results.  
-- resnet_results analysis.py: Confusion matrix, Precision, Recall and F1 score are obtained.  
  
***- efficientnet_realsize: efficientnet-b0.***    
-- model.py:Model structure.  
-- realsize_efficientnet.py:Run to get training and test results.  
-- realsize_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
-- efficientnet-b0.pth: The weight of imagenet.  
-- 400-300-2_train.txt: Training Set Directory.  
-- 400-300-2_test.txt: Test Set Directory.  


***- Pyramid of features: All of the features pyramid model,the backbone model is efficientnet-b0.***  
  
**-- SSD:Single Shot MultiBox Detector (SSD).** 
--- model_Fip.py:Model structure.  
--- efficientnet_chidu_Fip.py:Run to get training and test results.  
--- chidu_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
  
**-- FPN:Feature Pyramid Network (FPN).**  
--- model_FPN2.py:Model structure.  
--- efficientnet_FPN2.py:Run to get training and test results.  
--- FPN_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
  
**-- FPN_cat:Feature Pyramid Network (FPN),Use concatenation instead of addition.**  
--- model_FPN_cat.py:Model structure.  
--- efficientnet_FPN_cat.py:Run to get training and test results.  
--- FPN_cat_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
  
**-- PANet:Path Aggregation Network (PANet).**
--- model_PANet.py:Model structure.  
--- efficientnet_PANet.py:Run to get training and test results.  
--- PANet_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
  
**-- PANet_cat:Path Aggregation Network (PANet),Use concatenation instead of addition.**   
--- model_PANet_cat.py:Model structure.  
--- efficientnet_PANet_cat.py:Run to get training and test results.    

**-- BiFPN：Bidirectional Feature Pyramid Network(BiFPN).**  
--- model_BiFPN.py:Model structure.  
--- efficientnet_BiFPN.py:Run to get training and test results.  
--- BiFPN_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  
  
**-- FPN_cat_zeyou_vote:Our Proposed model.**   
--- model_FPN_cat_zeyou_vote.py:Model structure.  
--- efficientnet_FPN_cat_zeyou_vote.py:Run to get training and test results.  
--- FPN_cat_zeyou_vote_results analysis.py:Confusion matrix, Precision, Recall and F1 score are obtained.  

**- gram_cam_picture：To generate gradient‐weighted class activation mapping.**  
-- grad_cam:Where the code 'main_aa_cnn.py'represents the generated grad_cam and the code 'model_aa.py' represents the model structure.  
 


 
