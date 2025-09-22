# Computer Vision이란?

시각은 인간의 감각 정보의 대부분을 차지한다. 이러한 시각 정보를 "처리"하기 위해 절반 가량의 두뇌 자원이 사용된다.

*사실, 우리의 눈이 너무 구려서 인공지능이 pre-processing하듯이 떨림과 혈관을 제거해 주어야 하는데, 여기서 많은 자원이 사용된다! 인공 눈이 만들어진다면 우리의 뇌가 조금 더 자유로워지지 않을까.

이러한 시각 인지와 관련된 것들을 컴퓨터로 처리하는 것을 Computer Vision이라고 한다. 이를 위해, 단순히 컴퓨터에 대한 이해 뿐만 아니라 "인간의 시각 인지"에 대한 이해도 동반되어야 한다.

하지만 기준이 되어야 할 인간의 시각 인지부터가 완벽하지 않기 때문에, 컴퓨터 비전은 난해한 문제가 된다.

최근에는 이미지 분석뿐만 아니라, 거꾸로 이미지를 만드는 모델들도 나오고 있다. 

과거에는 정보를 통해 이미지를 만드는 Computer Graphics, Rendering에 반대되는 개념으로 이미지를 통해 정보를 추론하는 것을 Computer Vision, Inverse rendering이라고 했다.

현재는 Inverse rendering 뿐만 아니라 Rendering과 Img 2 Img process 또한 Computer Vision의 한 분야로 취급한다.

## 이미지 생성

컴퓨터는 카메라와 같은 도구를 통해 3D 세상을 2D로 project한다. (물론 인간의 눈과, 최근 카메라들은 여러 2D image를 통해 세상을 3D로 인식하는 듯 하다) 이렇게 투영된 이미지를 디지털화하여 우리가 사용하는 디지털 이미지가 완성된다. 디지털화 과정은 카메라 회사마다 다르다..

이미지 데이터는 위치, 값으로 이루어진다. Grayscale의 경우에는 하나의 채널만 사용하지만,

일반적으로 사용하는 우리의 RGB 이미지는 3개의 채널을 사용하고, 채도, 명도 등을 활용하는 다른 기법을 사용하는 이미지는 채널 구조가 다를 수 있다.



## CV 강의 개요

이번 주 강의는 이러한 Computer Vision의 발전사와, 데이터를 늘려 학습을 돕는 Data augmentation과 경량화와 self-supervised learning을 위한 knowledge distillation과 cv task를 다룬다.

### CV Tasks

#### Image segmentation

Object dection: 물체 인식 - 네모박스!

Segmentation: 물체 인식 - 픽셀 단위로 labeling

Instance segmentation : 서로 다른 인스턴스들을 찾아냄.

Semantic segmentation : "같은 클래스"의 인스턴스들을 묶음.

-> Panoptic segmentation : 모든 인스턴스들을 찾아내고, 그 클래스까지 분류한다!

#### Computation photography: 떨림 제거, 채도 조정, 화질 개선 등등..

#### 3D understanding: "이미지" to 3D

#### Generative model: 생성

Text - to - texture: 텍스쳐 입히기

3D human: 사람 모델링

# CV의 발전사
