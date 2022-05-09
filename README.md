
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


