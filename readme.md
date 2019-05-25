文件夹结构：  
├── data/  
│   ├── __init__.py  
│   ├── TestDatasets.py  ----测试数据准备  Test data preparation   
│   └── TrainDatasets.py  ----训练、验证数据准备  Train data preparation    
├── input/  ----数据集（限于大小并未导入数据）  Data set (limited to size does not import data)  
│   ├── test/  
│   ├── train/  
│   │ 	├── images/  
│   │ 	├── masks/  
│   ├── depths.csv  
│   ├── sample_submission.csv  
│   └── train.csv  
├── models/  
│   ├── __init__.py  
│   ├── TGS_UNet_ResNet34.py  ----基于ResNet34的Unet模型  Unet model based on ResNet34  
│   └── FocalLoss.py  ----Focalloss损失函数  
└── utils/  
│   ├── __init__.py    
│   ├── RLE.py  ----掩码图片的RLE编码与解码  RLE encoding and decoding of mask pictures  
│   └── Train.py  ----训练  Train  
├── main_test.ipynb  ----测试模型的主函数  Test the main function of the model  
├── main_train.ipynb   ----训练模型的主函数  The main function of the training model  
├── README.md  
