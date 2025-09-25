# Augmenatation

데이터셋은 "모든"데이터를 얻을 수 없고, 얻는 과정이 독립추출이기 어렵기 때문에 bias된 데이터를 얻게 되고,

이러한 데이터를 사용하게 되면 실전 데이터에서 robust하지 않게 동작할 수 있다.

예를 들어, 해바라기를 해가 떴을 때만 찍으면 어두운 해바라기는 제대로 판별하기 어려울 수 있다.

따라서, 이러한 gap을 줄이기 위해 데이터를 이리저리 흔들어서 실제 데이터 domain처럼 만들어 주는 것을 Augmentation이라고 한다.

## Technique - 정적

Crop: 짜르기

Shear: 기울이기-평행사변형꼴.

Brightness: 밝기 조절

Perspective Trnasformation: 원근 변환. 사진을 사다리꼴처럼 기울인다..

Rotate : 돌리기

flip : 뒤집기

Mixing: 이미지 두 개를 합친다!!! 이렇게 합친 이미지는 라벨도 그만큼 섞어 준다. one-hot label이 아니게 된다.

mixup: Beyond Empirical Risk Minimization: https://arxiv.org/abs/1710.09412 

여기서는 모든 pixel을 섞어 준다는 개념이고,

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features: https://arxiv.org/abs/1905.04899

여기서는 개 하반신, 고양이 상반신처럼 crop해서 붙이는 방식이다.

이러한 단순한 조합들이 생각 외로 Test Accuracy를 높여 준다.

Copy-paste: Segmentation된 부분들을 다른 데이터 위에 그려넣어 데이터를 만들어 준다.

## Learning-based Video Motion Magnification

copy-and paste를 통해 Augmentatiopn된 데이터를 만들고, (Object + Background)

Motion을 추가하기 위해서 단순히 Translation만 사용한다.(Random한 방향으로)

이렇게 "움직임"을 학습하면, 모션의 크기를 크게 하거나 하면서 흔들림, 이동 등을 과장할  수 있다.
