# RD4AD 프롬프트 

> 대상 오픈소스: RD4AD(Reverse Distillation from One-Class Embedding for Anomaly Detection)  
> 실행 환경: Google Colab  
> 데이터셋: MVTec AD `bottle` class  


## 1\. 오픈소스 코드 준비

### Prompt 1-1. RD4AD 실행 준비

Prompt

RD4AD GitHub repository를 Google Colab에서 실행하려고 해.
repository를 clone하고 작업 디렉토리로 이동하는 코드를 작성해줘.

AI 요약

Colab 환경에서 GitHub repository를 clone한 뒤 `/content/RD4AD`로 이동하도록 안내하였다.

```python
!git clone https://github.com/hq-deng/RD4AD.git
%cd RD4AD
```

실행 후 `dataset.py`, `de\_resnet.py`, `main.py`, `resnet.py`, `test.py` 등이 포함된 RD4AD 코드 구조를 확인하였다.


### Prompt 1-2. RD4AD 파일 구조 확인

Prompt

clone한 RD4AD 폴더 안에 어떤 파일들이 있는지 확인하고, 각 파일이 어떤 역할을 하는지 간단히 설명해줘.

AI 요약

`!ls` 명령으로 repository 내부 파일을 확인하였다.

주요 파일의 역할은 다음과 같이 정리하였다.

* `main.py`: RD4AD 학습 실행 파일
* `test.py`: evaluation 및 metric 계산 코드
* `dataset.py`: MVTec 데이터셋 로딩 관련 코드
* `resnet.py`: encoder 역할의 ResNet 계열 모델 정의
* `de\_resnet.py`: decoder 모델 정의
* `checkpoints/`: 학습된 weight 저장 위치


## 2\. Google Drive 및 데이터셋 준비

### Prompt 2-1. Google Drive mount

Prompt

MVTec AD `bottle.tar.xz` 파일이 Google Drive에 있어.
Colab에서 Drive를 mount하고 파일이 있는지 확인하는 코드를 작성해줘.

AI 요약

Google Drive를 mount한 뒤, `/content/drive/MyDrive` 내부 파일 목록을 확인하도록 코드를 작성하였다.

```python
from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive
```

실행 결과 Google Drive 안에 `bottle.tar.xz` 파일이 존재함을 확인하였다.


### Prompt 2-2. MVTec bottle 데이터셋 압축 해제

Prompt

Google Drive에 있는 `bottle.tar.xz`를 RD4AD의 `mvtec` 폴더 아래에 압축 해제하고 싶어.
필요한 폴더 생성과 압축 해제 코드를 작성해줘.

AI 요약

RD4AD 내부에 `mvtec` 폴더를 만들고, `tar -xJf` 명령으로 `bottle.tar.xz` 파일을 압축 해제하도록 안내하였다.

```python
!mkdir -p /content/RD4AD/mvtec
!tar -xJf /content/drive/MyDrive/bottle.tar.xz -C /content/RD4AD/mvtec
```

압축 해제 후 다음 구조가 확인되었다.

```text
/content/RD4AD/mvtec/bottle/
  ground\_truth/
  test/
  train/
  license.txt
  readme.txt
```


### Prompt 2-3. 데이터셋 폴더 구조 확인

Prompt

압축 해제된 MVTec AD bottle 데이터셋이 RD4AD 코드에서 사용할 수 있는 구조인지 확인하고 싶어.  
폴더 구조 확인 명령을 작성해줘.

AI 요약

`mvtec` 폴더와 `bottle` 폴더 내부를 확인하는 명령을 작성하였다.

```python
!ls /content/RD4AD/mvtec
!ls /content/RD4AD/mvtec/bottle
```

`train`, `test`, `ground\_truth` 폴더가 존재하므로 RD4AD 학습 및 평가에 필요한 기본 데이터셋 구조가 준비된 것으로 판단하였다.


## 3\. 필수 package 설치

### Prompt 3-1. 필요한 라이브러리 설치

Prompt

RD4AD를 Colab에서 실행할 때 필요한 Python package를 설치하고 싶어.  
`scikit-learn`, `scipy`, `pillow` 설치 코드를 작성해줘.

AI 요약

다음 명령을 통해 필요한 package를 설치하였다.

```python
!pip install scikit-learn scipy pillow
```

Colab에는 해당 package들이 이미 설치되어 있었으며, 기존 설치 버전을 그대로 사용하였다.


## 4\. 학습 대상 class 수정

### Prompt 4-1. main.py에서 모든 class가 아니라 bottle class만 학습

Prompt

RD4AD의 `main.py`는 MVTec AD 전체 class를 순서대로 학습하도록 되어 있어.  
이번 실험에서는 `bottle` class만 사용하므로, `main.py`의 실행 부분을 `train('bottle')`만 수행하도록 수정하는 코드를 작성해줘.

AI 요약
`main.py`의 `if \_\_name\_\_ == '\_\_main\_\_':` 아래 class list 반복 실행 부분을 `train('bottle')`로 바꾸는 코드를 작성하였다.

```python
from pathlib import Path

p = Path("/content/RD4AD/main.py")
txt = p.read\_text()

old = """if \_\_name\_\_ == '\_\_main\_\_':
 setup\_seed(111)
 item\_list = \['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
 'transistor', 'metal\_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
 for i in item\_list:
  train(i)"""

new = """if \_\_name\_\_ == '\_\_main\_\_':
 setup\_seed(111)
 train('bottle')"""

txt = txt.replace(old, new)
p.write\_text(txt)
```

이를 통해 전체 class를 모두 학습하지 않고, 준비된 `bottle` 데이터셋만 대상으로 실험을 진행하도록 수정하였다.


### Prompt 4-2. checkpoints 폴더 생성

Prompt

RD4AD 학습 결과 weight를 저장하기 위한 `checkpoints` 폴더가 필요해.  
없으면 자동으로 생성하는 코드를 작성해줘.

AI 요약

다음 명령으로 checkpoint 저장 폴더를 생성하였다.

```python
!mkdir -p /content/RD4AD/checkpoints
```


## 5\. 경로 및 checkpoint 이름 수정

### Prompt 5-1. test.py 데이터셋 경로 수정

Prompt

`test.py`에서 MVTec 데이터셋 경로가 `../mvtec/`로 되어 있는데, 현재 데이터셋은 `/content/RD4AD/mvtec/` 아래에 있어.  
현재 실행 위치 기준으로 `./mvtec/`를 사용하도록 코드를 수정해줘.

AI 요약

`test.py`에서 test dataset path를 다음과 같이 수정하였다.

```python
from pathlib import Path

p = Path("/content/RD4AD/test.py")
txt = p.read\_text()
txt = txt.replace("test\_path = '../mvtec/' + \_class\_", "test\_path = './mvtec/' + \_class\_")
p.write\_text(txt)

print("test 경로 고침")
```

이를 통해 `/content/RD4AD`에서 실행했을 때 `./mvtec/bottle` 경로를 정상적으로 읽도록 수정하였다.


### Prompt 5-2. checkpoint 파일 이름 수정

Prompt

RD4AD의 `test.py`에서 checkpoint 이름을 `rm\_1105\_wres50\_ff\_mm\_bottle.pth` 형태로 찾고 있어.  
하지만 이번 학습에서는 `wres50\_bottle.pth` 이름으로 저장되도록 맞추고 싶어.  
`test.py`의 checkpoint load 경로를 수정하는 코드를 작성해줘.

AI 요약

`test.py`에서 checkpoint path를 `wres50\_bottle.pth` 형식으로 읽도록 수정하였다.

```python
from pathlib import Path

p = Path("/content/RD4AD/test.py")
txt = p.read\_text()
txt = txt.replace(
    "ckp\_path = './checkpoints/' + 'rm\_1105\_wres50\_ff\_mm\_' + \_class\_ + '.pth'",
    "ckp\_path = './checkpoints/' + 'wres50\_' + \_class\_ + '.pth'"
)
p.write\_text(txt)

print("체크포인트 이름 고침")
```

학습 결과 저장 파일과 평가 코드에서 불러오는 파일명을 일치시켰다.


## 6\. 실행 오류 확인 및 수정

### Prompt 6-1. carpet class 경로 오류 해결

Prompt

`python main.py`를 실행했더니 `carpet` class부터 학습하려고 하면서 데이터셋 경로 오류가 발생했어.  
현재는 `bottle` 데이터셋만 준비되어 있는데, 왜 이런 문제가 생기고 어떻게 고쳐야 하는지 알려줘.

AI 요약
오류 원인은 `main.py`가 여전히 전체 class list를 반복 실행하고 있었기 때문이라고 분석하였어.  
따라서 `if \_\_name\_\_ == '\_\_main\_\_':` 부분을 다시 확인하고, 아래처럼 `train('bottle')`만 실행하도록 수정하였다.

```python
from pathlib import Path

p = Path("/content/RD4AD/main.py")
txt = p.read\_text()

start = txt.find("if \_\_name\_\_ == '\_\_main\_\_':")
txt = txt\[:start] + """if \_\_name\_\_ == '\_\_main\_\_':

    setup\_seed(111)
    train('bottle')
"""

p.write\_text(txt)
```

수정 후 `tail` 명령으로 마지막 실행부가 정상적으로 바뀌었는지 확인하였다.

```python
!tail -n 10 /content/RD4AD/main.py
```

### Prompt 6-2. pandas append 오류 해결

Prompt

RD4AD 실행 중 `df.append()` 관련 오류가 발생했어.  
최신 pandas에서 `DataFrame.append()`가 제거된 것으로 보이는데, 이를 어떻게 수정해야 하는지 알려줘.

AI 요약

최신 pandas에서는 `df.append()` 대신 `df.loc\[len(df)] = ...` 방식으로 행을 추가해야 한다고 안내하였다.  
`test.py`의 기존 코드를 다음과 같이 수정하였다.

```python
from pathlib import Path

p = Path("/content/RD4AD/test.py")
txt = p.read\_text()
txt = txt.replace(
    'df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore\_index=True)',
    'df.loc\[len(df)] = {"pro": mean(pros), "fpr": fpr, "threshold": th}'
)
p.write\_text(txt)

print("append 에러 수정 완료")
```

이 수정은 library version 변화에 따른 compatibility patch로 정리하였다.



## 7\. RD4AD 학습 실행

### Prompt 7-1. bottle class 학습 실행

Prompt

모든 경로 수정과 compatibility patch를 적용했어.  
이제 RD4AD를 `bottle` class에 대해 학습 실행하는 명령을 작성해줘.

AI 요약

RD4AD 폴더로 이동한 뒤 `main.py`를 실행하였다.

```python
%cd /content/RD4AD
!python main.py
```

10 epoch마다 Pixel AUROC, Sample AUROC, Pixel AUPRO가 출력되었다.


### Prompt 7-2. 학습 결과 metric 정리

Prompt

RD4AD bottle class 학습 로그에서 10 epoch 단위로 출력된 metric을 표로 정리해줘.  
Pixel AUROC, Sample AUROC, Pixel AUPRO를 기준으로 보면 돼.

AI 요약

학습 로그에서 확인된 결과는 다음과 같다.

|Epoch|Pixel AUROC|Sample AUROC|Pixel AUPRO|
|-:|-:|-:|-:|
|10|0.982|0.989|0.953|
|20|0.987|0.998|0.967|
|30|0.987|0.999|0.967|
|40|0.987|1.000|0.966|
|50|0.987|1.000|0.966|

최종적으로 `bottle` class에 대해 Pixel AUROC 0.987, Sample AUROC 1.000, Pixel AUPRO 0.966 수준의 결과가 확인되었다.


### Prompt 7-3. checkpoint 저장 확인

Prompt

학습이 끝난 뒤 checkpoint가 정상적으로 저장되었는지 확인하는 코드를 작성해줘.

AI 요약

`checkpoints` 폴더 내부를 확인하였다.

```python
!ls /content/RD4AD/checkpoints
```

실행 결과 `wres50\_bottle.pth` 파일이 생성된 것을 확인하였다.  
따라서 학습된 bottleneck 및 decoder weight가 정상적으로 저장된 것으로 판단하였다.


## 8\. 테스트 이미지 및 ground truth mask 확인

### Prompt 8-1. bottle test 폴더 구조 확인

Prompt

MVTec bottle test 데이터셋에 어떤 defect type이 있는지 확인하고 싶어.  
test 폴더 내부를 확인하는 명령을 작성해줘.

AI 요약

다음 명령으로 test 폴더 내부 defect type을 확인하였다.

```python
!ls /content/RD4AD/mvtec/bottle/test
```

확인된 defect type은 다음과 같다.

```text
broken\_large
broken\_small
contamination
good
```


### Prompt 8-2. 특정 defect image 확인

Prompt

`broken\_large` defect type의 이미지 파일 목록을 확인하고, 그중 `000.png` 이미지를 화면에 출력하는 코드를 작성해줘.

AI 요약

`broken\_large` 폴더 내부를 확인한 뒤, PIL과 matplotlib을 이용해 원본 이미지를 시각화하였다.

```python
!ls /content/RD4AD/mvtec/bottle/test/broken\_large
```

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/content/RD4AD/mvtec/bottle/test/broken\_large/000.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

이를 통해 실제 test image를 육안으로 확인하였다.


### Prompt 8-3. ground truth mask 확인

Prompt

`broken\_large/000.png`에 대응하는 ground truth mask를 열어보고 싶어.  
mask 이미지를 흑백으로 출력하는 코드를 작성해줘.

AI 요약

`ground\_truth/broken\_large/000\_mask.png` 파일을 열고 grayscale로 시각화하였다.

```python
from PIL import Image
import matplotlib.pyplot as plt

mask = Image.open('/content/RD4AD/mvtec/bottle/ground\_truth/broken\_large/000\_mask.png')
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()
```

이를 통해 anomaly map 결과와 비교할 정답 mask를 확인하였다.


## 9\. 학습된 모델을 이용한 anomaly map 시각화

### Prompt 9-1. RD4AD checkpoint load 및 모델 구성

Prompt

학습된 `wres50\_bottle.pth` checkpoint를 불러와서 RD4AD encoder, bottleneck, decoder를 구성하고 싶어.  
`wide\_resnet50\_2` encoder와 `de\_wide\_resnet50\_2` decoder를 사용하는 코드를 작성해줘.

AI 요약

RD4AD의 `resnet.py`, `de\_resnet.py`에서 모델을 import하고, 학습된 checkpoint에서 `bn`과 `decoder` weight를 불러오는 코드를 작성하였다.

```python
import torch
from resnet import wide\_resnet50\_2
from de\_resnet import de\_wide\_resnet50\_2

device = 'cuda' if torch.cuda.is\_available() else 'cpu'

encoder, bn = wide\_resnet50\_2(pretrained=False)
decoder = de\_wide\_resnet50\_2(pretrained=False)

encoder = encoder.to(device)
bn = bn.to(device)
decoder = decoder.to(device)

ckpt = torch.load('/content/RD4AD/checkpoints/wres50\_bottle.pth', map\_location=device)
bn.load\_state\_dict(ckpt\['bn'])
decoder.load\_state\_dict(ckpt\['decoder'])

encoder.eval()
bn.eval()
decoder.eval()
```

RD4AD 구조상 encoder는 feature extraction 역할을 하고, decoder는 encoder feature를 복원하는 역할을 한다.  
복원 feature와 원본 feature 간 차이를 이용해 anomaly score를 계산한다.


### Prompt 9-2. 이미지 전처리 코드 작성

Prompt

MVTec bottle 이미지를 RD4AD 모델에 입력하기 위해 256×256 resize, tensor 변환, ImageNet normalization을 적용하는 전처리 코드를 작성해줘.

AI 요약

`torchvision.transforms`를 이용해 이미지 전처리 pipeline을 구성하였다.

```python
from torchvision import transforms

transform = transforms.Compose(\[
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=\[0.485, 0.456, 0.406],
                         std=\[0.229, 0.224, 0.225])
])
```

이 전처리는 RD4AD에서 사용하는 ResNet 기반 encoder 입력 형식에 맞추기 위한 과정이다.


### Prompt 9-3. cosine similarity 기반 anomaly map 계산

Prompt

RD4AD에서 encoder feature와 decoder output feature의 차이를 이용해 anomaly map을 계산하고 싶어.  
각 feature scale마다 `1 - cosine\_similarity`를 계산하고, 256×256 크기로 보간한 뒤 합산하는 코드를 작성해줘.

AI 요약
다음 방식으로 anomaly map을 계산하였다.

```python
import torch.nn.functional as F

with torch.no\_grad():
    inputs = encoder(x)
    outputs = decoder(bn(inputs))

    anomaly\_map = torch.zeros((1, 1, 256, 256)).to(device)
    for i in range(len(inputs)):
        a\_map = 1 - F.cosine\_similarity(inputs\[i], outputs\[i], dim=1)
        a\_map = a\_map.unsqueeze(1)
        a\_map = F.interpolate(a\_map, size=(256, 256), mode='bilinear', align\_corners=False)
        anomaly\_map += a\_map

    anomaly\_map = anomaly\_map.squeeze().cpu().numpy()
```

encoder feature와 decoder가 복원한 feature가 다를수록 anomaly score가 커지므로, `1 - cosine\_similarity` 값을 anomaly map으로 사용하였다.


### Prompt 9-4. anomaly map 단독 시각화

Prompt

계산한 anomaly map을 matplotlib으로 시각화하는 코드를 작성해줘.

AI 요약
다음 코드로 anomaly map을 출력하였다.

```python
plt.imshow(anomaly\_map)
plt.axis('off')
plt.show()
```

이를 통해 모델이 `broken\_large/000.png` 이미지에서 이상 영역을 어느 위치로 판단하는지 시각적으로 확인하였다.


### Prompt 9-5. 원본 이미지, GT mask, anomaly map 비교 시각화

Prompt

원본 이미지, ground truth mask, anomaly map을 한 화면에 나란히 비교하고 싶어.  
3개의 이미지를 subplot으로 출력하는 코드를 작성해줘.

AI 요약

원본 이미지, 정답 mask, 모델의 anomaly map을 1행 3열로 배치하여 비교하였다.

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Ground Truth Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(anomaly\_map)
plt.title('Anomaly Map')
plt.axis('off')

plt.show()
```

이를 통해 정량 metric뿐만 아니라, 실제 결함 위치를 모델이 어느 정도 반영하는지 정성적으로도 확인하였다.