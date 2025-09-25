# Segmentation & Detection

## Sementic segmentation 

각각의 픽셀을 카테고리로. 어떤 Instance인가?를 따지는 것은(서로 다른 자동차인지..)는 Instance segmentation에서 다룬다.

의료, 자율주행 등 다양한 CV 분야에 사용된다.

### FCN

CNN과 다르게, flatten을 하지 않고 마지막까지 CNN 단을 유지하여.. pixelwise하게 결과를 뽑아낸다.

그런데, 이런 경우 CNN에서 max pooling이나, padding 없는 convolution에 의해 이미지 사이즈가 줄어들기 때문에 "작은" 이미지가 나오게 된다.

하지만, 마지막 단의 레이어는 많은 정보를 담고 있지만, 이미 "합쳐진" 데이터이기 때문에 이것만으로 upsampling하기에는 무리가 있다.

<img width="1636" height="383" alt="image" src="https://github.com/user-attachments/assets/7ce82f10-454c-4bef-914f-5ac97672a303" />
https://arxiv.org/pdf/1411.4038

그래서, 마지막 커널을 통과한 값들 뿌남ㄴ 아니라 다른 값들도 사용한다.

그림에 나와 있는 FCN-32s는 마지막 단만 사용한 것으로, FCN-16s, FCN-8s 등은 그 전 단, 그 전 전 단까지 사용한 결과이다.

너무 앞쪽 단에는 feature extraction이 다 되어 있지 않을 수 있기 때문에 8x가 좋은 결과를 뱉는 것으로 보인다.

### U-net

<img width="1697" height="757" alt="image" src="https://github.com/user-attachments/assets/a3a43331-7b48-45b4-9e39-c20360521dc4" />
[ Ronneberger etal.,U-Net:ConvolutionalNetworksforBiomedical ImageSegmentation,MICCAI2015](https://arxiv.org/abs/1505.04597)

U-net은 FCN과 비슷하게, Contracting path(왼쪽)에서 절반씩 downsizing을 하면서 feature channel의 수를 두배로 늘리고,

*이때, upscaling을 할 때 "2배"의 이미지가 되기 때문에, downsizing을 할 때 홀수인 경우 사이즈가 이상해진다. 따라서, 커널의 사이즈 등을 조절해서 항상 downsizing을 할 때 짝수가 되도록 해 준다.

오른쪽의 Expanding path에서는 반대로 채널을 줄여 가며 이미지를 upscaling한다.

논문을 살펴보면.. 왼쪽과 오른쪽의 이미지 사이즈가 다르다. upscaling할 때 어떻게 그러면 이어붙이냐?고 하면..

그냥 가운데 중심으로 자른다고 한다.얼탱.

## Object Detection

Bounding box를 찾는 과정이다. (+Classification)

보통 박스 하나는 $(p_{class}, x_1,x_2,y_1,y_2)$ 꼴로 나타난다.

### Two-stage(R-CNN)

Bounding box를 먼저 찾고(Selective search를 사용한다. 이미지들로부터, 후보 이미지들을 모두 뽑아내는 Pre-trained된 모델이고, CNN 구조는 아니다)

그 Bounding box들을 동일 사이즈로 warp 한 후(CNN에 넣기 위해) CNN으로Clasification하는 두 단계로 이루어진 아키텍쳐이다.

기본적이 R-cnn은, 인풋 이미지에서 수많은 region 제안을 뽑고, 그 제안들에 대해 모두... CNN을 돌린다.

당연하겠지만, 무척 오래 걸린다.. 오래 걸리고, 데이터도 모자라서 그런지 SVM object classifier를 사용했다.

<img width="1372" height="707" alt="image" src="https://github.com/user-attachments/assets/381017fb-0dae-4583-961d-b095c6e24abb" />
https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3.html

Fast R-CNN은 "미리" 이미지를 통과시켜 RoI map를 투영할 feature map 만들어 놓고, proposal들을 그 위에 proejct한 후 max pooling을 통해 고정된 크기의 값을 뱉도록 한다. 이렇게 하면, fully connected layer에 넣기 위한 vector의 길이를 맞출 수 있다.

Fast R-CNN은 이러한 RoI pooling을 통해 2000번의 CNN 계산을 한 번으로 줄였다.

Faster R-cnn은 RPN을 통해 Region proposal마저 CNN으로 계산하게 했다. (이전의 Selective search는 굉장히 느린 작업이었다)

### One-stage(YOLO)

RoI pooling을 거치지 않고, 한 번에 classification과 bounding box를 찾는 방식이다.

Yolo는 이미지를 그리드로 나눈 뒤, 각각 Class probability map과 Bounding box(Confidence를 포함한)을 만든 뒤, 두 개를 합쳐 detection을 한다.

그리드 갯수 $S \times S$ , 그리드당 bounding box $B$ , 클래스 갯수 $C$ 에 대해,

$S \ times S \times B \times (C+5)$ 

크기의 output이 나온다! 5는 바운딩 박스의 크기와 confidence이다.

다라서 후처리가 필요하다.. confidence score를 기준으로 하든.. 좀 시간이 걸리고,

이렇게 후처리 끝나고 결정된 값들만 사용된다.

#### Focal loss

사실 대부분의 영역은 의미없는 값들이다. 대부분의 box들은 의미없는 배경만을 잡을 것이고, 중요한 정보가 아닌데도 너무 많은 숫자를 가지고 있어, 학습이 느려지는 문제가 있다.

그래서 크로스 엔트로피 대신, Focal Loss를 사용한다.

크로스 엔트로피에 $(1-p_t)^{\gamma}$ 를 곰한 값이다. ( $\gamma$ 는 hyperparameter)

이렇게 하면, 잘 판별된 결과들에 대한 loss가 굉장히 줄어들어, "확실히 배경이여!" 하는 부분에서 쓸모없이 학습이 진행되지 않게 한다. 

이를 활용한 것이 RetinaNet이다.

<img width="1727" height="537" alt="image" src="https://github.com/user-attachments/assets/9d657bea-07e7-473b-8117-189847ed348f" />
https://arxiv.org/abs/1708.02002

U-net처럼 추출하고, skip connection을 통해 합쳐 주는 것은 같지만, class와 box subnet을 나누어 object detection에 사용될 수 있게 하고, Focal loss를 통해 배경의 영향을 줄인다.

## Instance segmentation: Mask R-CNN

같은 클래스더라도, 다른 인스턴스인 경우 "다른 인스턴스임을 알리는 번호"를 매기는 작업까지 한다.

Mask R-CNN은 Faster R-CNN 구조의 classifier 단에 Mask FCN predictor를 더한 구조이다.

여기서 RoI pooling 대신 RoI align을 사용한다.

RoI pooling에서는 양자화된 feature map을 사용하는데, 여기서 양자화되어 있기 때문에 RoI와 격자의 크기가 맞지 않으면 버려지는 문제가 있다. 이러한 문제는 masking task에서는 치명적일 수 있기 때문에, 양자화를 하지 않고, RoI를 나눈 cell들에 대해 4개의 point를 잡아 Bilinear interpolation을 통해 Cell의 값을 추출한다. (중간의 값을 선형적으로 찾는 방법, 선형회귀라고 생각하면 된다)

여기서 Faster R-CNN에 Mask 대신 3D surface를 회귀시켜 3D 정보를 추출하는 Mesh R-CNN이나 DensePose R-CNN을 만들 수도 있다.

## Transformer-based

### DETR: End-to-End Object DEtection with TRansformers

Yolo 등의 모델에서, 의미없는 bounding box를 지우는 작업은 꽤나 시간이 걸린다. 그러한 문제를 해결하기 위해.. End-to-End 로 object detion을 하는 Transformer model을 DETR이라고 한다.

<img width="1527" height="413" alt="image" src="https://github.com/user-attachments/assets/aa11ea3f-6ca6-4b10-a3e6-672226730aa1" />

기본적으로 이미지의 feature들을 추출하기 위해 CNN을 사용하고, 여기에 positional encoding을 더해 transformer에 넣어 준다.

encoder의 경우는 동일한데, decoder의 경우에는 이전 output을 활용한 auto-regressive한 값이 아니라 학습되는 object queries를 대신 넣어 준다. 이렇게 decoder에서 나온 값들을 FFN을 통과해서, 정답과 예측 값이 1:1이 되도록 하는 bipartite matching을 구해 loss를 계산한다.

이때, 최적의 bipartite matching을 찾기 위해 matching loss와 box loss를 이용한 Hungarian algorithm 등을 사용한다.

### MaskFormer

<img width="1493" height="386" alt="image" src="https://github.com/user-attachments/assets/88df3db9-d51f-4ff5-8499-b80cae8d649d" />

<img width="1423" height="449" alt="image" src="https://github.com/user-attachments/assets/94459b4e-9d8b-4c3e-9803-cf6800265ba4" />

Maskformer에서는 Mask R-CNN처럼, Mask를 따로 학습하여 픽셀 단위 분류 대신 사용한다.

https://arxiv.org/abs/2107.06278

Backbone CNN에서의 feature를 Transformer에 넣고, 이미지 해상도로 decoding하여 다음 step의 계산을 준비한다.

transformer decoder의 결과를(Query가 학습되며 position도 학습한다) MLP를 거쳐서 class prediction과 mask embedding로 나눈다.

이전에 계산한 pixel embedding에 mask embedding을(softmax를 거쳐) dot-product해준 것을 mask prediction으로 사용한다.

N개의 Query가 있기 때문에, class predictions(class 수 +1(아무것도 아님))와 mask prediction은 N개가 존재한다. 이 두 개를 합쳐 예측을 수행하고, loss를 구한다.



### Uni-DVPS

Backbone에서 Transformer Decoder의 output을 Transformer decoder의 key-value로 하고, Depth를 따로 계산하는 MLP를 추가해 주는 모델이다.

Pixel decoder에서 나온 값을  Mask, Depth 계산에 사용할 수 있게 옮겨 주는 Feature gate가 있어, 이를 mask와 Depth에도 사용하는 모델이다.

이때, Query matching을 통해 Object를 추적할 수 있다. 영상에서, 같은 object의 경우 비슷한 query를 갖기 때문에 이를 통해 유사도를 계산해 같은 object인지 확인할 수 있다.

## SAM, Grounded SAM

<img width="500" height="358" alt="image" src="https://github.com/user-attachments/assets/2e66bd73-2d3a-4b57-8173-dda304d08ce3" />

[SegmentAnything,](https://arxiv.org/abs/2304.02643#:~:text=We%20introduce%20the%20Segment%20Anything%20%28SA%29%20project%3A%20a,masks%20on%2011M%20licensed%20and%20privacy%20respecting%20images.)

image에, 우리가 원하는 추가적인 프롬프트(포인트, 텍스트, 마스크, 박스 등등..)을 넣어서 masking을 하는 마법같은 모델이다.


<img width="922" height="1397" alt="image" src="https://github.com/user-attachments/assets/09b62421-6f1d-43e0-b772-8a436c865460" />

<img width="581" height="788" alt="image" src="https://github.com/user-attachments/assets/4ea19de9-b469-46c0-b294-27e4ad761976" />


https://maucher.pages.mi.hdm-stuttgart.de/orbook/deeplearning/SAM.html


ViT Encoder를 사용하고, 임베딩된 데이터에 mask를 추가한 뒤 prompt encoder를 통과시킨 prompt와 mask image를 mask decoder에 넣는데,

prompt를 query로 받고, mask image를 key-value로 하는 token to image attention 뿐만 아니라 image to token을 하는 decoder 두 단을 쌓는다. 

image의 positional embedding과 prompt embedding은 다음 단에도 더해져서 들어간다는 것을 기억하자.

이렇게 image to token쪽을 통과한 값은 upscale와 CNN을 거쳐, token to image가 final attention을 지난 값의 mask token을 MLP를 통과시켜서 product해서 mask를 만들게 되고, iou token은 따로 MLP를 통과시켜 confidence score로 받게 된다.

내부 구조는 position encoding이 계속 더해지고, key, value를 구하는 과정에서 이것저것 더해져서 조금 복잡하다..

이 SAM을 학습시키기 위해 데이터 생성 과정이 재밌는데, 처음에는 조그마한 라벨링된 데이터로 시작해서,

어느 정도 이후에는 모델이 뱉은 label을 사람이 수정해서 조금 더 쉽게 labeling하고ㅗ,

그 다음에는 모델이 "확실하다"고 하는 것들은 그냥 모델의 라벨링을 믿고 나머지를 사람들이 labeling하고,

그 다음에는 완전히 자동적으로 모델이 label을 추가하며 자동적으로 학습하게 했다.

특정 데이터에는 안 좋은 결과가 나왔지만, 대부분의 결과에 좋은 데이터가 나왔다. (게다가, 이것은 특정 데이터로 fine-tuning하지 않은 모델이다!)

Grounded-SAM은 G-DINO, detection을 한 뒤 context를 주는 모델과 결합시켜 Segmentation에 더해 Open-Vocab detection까지 추가한 모델이다.
