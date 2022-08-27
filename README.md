# Important Notice
This published code is usesed in paper: "Weakly supervised training for eye fundus lesion segmentation in patients with diabetic retinopathy"

It can be only used for non-commercial use such as scientific research and education. Please feel free to contact with us if there is any problem at yuli/at/bjut.edu.cn

Do not forget to reference our paper in your publication by:
Yi Li, Meilong Zhu, Guangmin Sun et al., Weakly supervised training for eye fundus lesion segmentation in patients with diabetic retinopathy, Math. Biosci. Eng.
,2022, Mar 24;19(5):5293-5311. doi: 10.3934/mbe.2022248.

Thank you for your interest :)

Lab of Nerual Networks and Image Recognition
Beijing University of Technology
Contact: yuli@bjut.edu.cn

# RAUNet 
RAUNet network for fundus image segmentation
## data preparation
structure of project
'''
  --project
  	main.py
  	 --data
   		--train
   		--val

partially open dataset：OIA-DDR:https://github.com/nkicsl/DDR-dataset
                                       Messidor：http://www.adcis.net/en/third-party/messidor

## training
'''
python main.py train



## testing
load the weight

python main.py test --ckpt=weights_xx.pth


----

## Data samples
Data files include:
'''
  --project
  	main.py
  	 --data
   		--train
   		--val
      
partially open dataset: OIA-DDR:https://github.com/nkicsl/DDR-dataset
                          Messidor：http://www.adcis.net/en/third-party/messidor

## Model Training
'''
python main.py train


## Model Testing
Load the model weights
'''
python main.py test --ckpt=weights_xx.pth


