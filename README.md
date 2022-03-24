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
'''
partially open dataset：OIA-DDR:https://github.com/nkicsl/DDR-dataset
                                       Messidor：http://www.adcis.net/en/third-party/messidor

## training
'''
python main.py train

'''

## testing
load the weight
'''
python main.py test --ckpt=weights_xx.pth

'''
----

## 数据准备
项目文件分布如下
'''
  --project
  	main.py
  	 --data
   		--train
   		--val
'''
部分公开数据集: OIA-DDR:https://github.com/nkicsl/DDR-dataset
                          Messidor：http://www.adcis.net/en/third-party/messidor

## 模型训练
'''
python main.py train

'''

## 测试模型训练
加载权重
'''
python main.py test --ckpt=weights_xx.pth

'''
