Binary Classification
=====================
실험주제
-------
다양한 classification Model을 이진분류 개 고양이 분류 데이터를 활용하여 동일한 하이퍼 파라미터 상에서
성능을 비교하고 상위 2개 모델의 Attention SE Block을 추가하여 성능 향상의 여부를 판별.

실험환경
--------
* Google Colab Gpu 을 사용하였음.
* Colab 내장 라이브러리 PyTorch '1.5.0+cu101' 환경

DataSet
-------
> Image Example   
   
<img src="/workspace/binary_classification/image/1.JPG" width="80%" height="80%" title="img1" alt="img1"></img>   

> 분류 그래프   
   
<img src="/workspace/binary_classification/image/2.JPG" width="80%" height="80%" title="img2" alt="img2"></img>   

* 데이터셋은 다음의 링크를 참조 : <https://www.kaggle.com/c/dogs-vs-cats/data>   
* 데이터 포맷 형식 : jpg   
* 데이터 분류 방식 : Download 한 Daataset 내부 Train dataset 25000장 을 배열 슬라이스를 통하여 Train iamge 17000장, vaildation image 4000장, test image 4000장으로 나누어서 실험하였다.(분류 그래프 참조)
* 이미지 해상도 :
* class : 개, 고양이

Augmentation
------------
* Train Dataset :Resize, RandomHorizontalFlip
* Validation, Test Dataset : Resize

Hyperparameter
--------------
* epoch : 200 
* image size : height = 224, width = 224, channel = 3 로 변환
* optimizer : SGD ( decay = 5e-4, momentum = 0.9 )
* learning rate : 1e-5

Training
----------
```
python main.py -h
modelnum list
------------------------------
1: Vgg11
2: Vgg13
3: Vgg16
4: Vgg19
5: Resnet18
6: Resnet34
7: Resnet50
8: Resnet101
9: Resnet152
10:DenseNet121
11:DenseNet169
12:DenseNet201
13:DenseNet161(growth_rate = 48)
------------------------------
usage: main.py [-h] [-se] [-show] modelnum lr epochs

Learn by Modeling Dog Cat DataSet

positional arguments:
  modelnum    Select your model number
  lr          Select opimizer learning rate
  epochs      Select train epochs

optional arguments:
  -h, --help  show this help message and exit
  -se         Put the selayer in the model.
  -show       show to model Archtecture

```
실험결과
-------
### Test Dataset의 Accuracy 비교

|Model|기존모델 Accuarcy|+SE Accuarcy|
|---|---|---|
|VGG11|0.9152|0.848|
|VGG16|0.919|0.9082|
|Resnet18|0.869|0.8697|
|Resnet50|0.7592| __|
|Densenet121|0.87|0.897|

***
### Training & Validation Graph, Confusion Matrix
#### VGG11    
    
> 기존모델   
   
<img src="/workspace/binary_classification/image/3.jpg" width="80%" height="80%" title="img3" alt="img3"></img>   
> SEVGG11   

<img src="/workspace/binary_classification/image/8.jpg" width="80%" height="80%" title="img3" alt="img3"></img>   

#### VGG16
> 기존모델   
      
<img src="/workspace/binary_classification/image/4.jpg" width="80%" height="80%" title="img4" alt="img5"></img>   
> SEVGG16   
   
<img src="/workspace/binary_classification/image/9.jpg" width="80%" height="80%" title="img4" alt="img5"></img>     
    
#### Resnet18  
> 기존모델   
         
<img src="/workspace/binary_classification/image/5.jpg" width="80%" height="80%" title="img5" alt="img5"></img>   
> SEResNet18   
      
<img src="/workspace/binary_classification/image/10.jpg" width="80%" height="80%" title="img5" alt="img5"></img>     

#### Resnet50 
> 기존모델   
        
<img src="/workspace/binary_classification/image/6.jpg" width="80%" height="80%" title="img5" alt="img5"></img>   

#### DenseNet121
> 기존모델   
           
<img src="/workspace/binary_classification/image/7.jpg" width="80%" height="80%" title="img5" alt="img5"></img>     

> SEDenseNet121   
   
<img src="/workspace/binary_classification/image/11.jpg" width="80%" height="80%" title="img5" alt="img5"></img>    


