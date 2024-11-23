# 🛌 **낙상 감지 프로젝트**

![Fall Detection](https://github.com/KangheeCHO-git/fall-down/blob/main/fall_down.png)

* 피실험자 낙상 이미지를 이용하여 yolov8s의 사람 객체 탐지 및 bbox 추출 기능을 학습/검증함

    (기존 Label json 데이터셋을 yolo 학습용 txt 데이터셋으로 변환함) 

* 피실험자 낙상 동영상에서 mediapipe를 이용하여 사람 객체 탐지 및 11개의 관절 좌표값을 추출함

    (mediapipe는 label 데이터셋이 없으므로 성능 검증이 불가능함, 대신 yolo에서 탐지된 사람 객체 bbox 내부에서만 mediapipe 관절 좌표값을 잡기로 함)

* 피실험자 낙상 동영상에서 추출한 yolo 및 mediapipe의 시계열 데이터를 이용하여 GRU 모델에 넣을 input data를 결정함

* 기존 동영상 label에 기록된 낙상 중인 시점을 이용하여 피실험자 상태(정상/낙상 중/낙상 후)를 정답으로 classification함


❗ **GRU 모델에 넣을 input data에 따라 낙상 감지 모델의 성능이 향상됨을 확인하고, LSTM 모델과의 성능 차이를 비교하는 것이 프로젝트의 목표**

❗ **GRU 모델과 달리 낙상 속도를 입력 데이터로 사용하지 않고, 시간적 의존성이 낮은 bbox 비율과 Mediapipe 관절 좌표값만을 input data로 학습한 CNN 모델의 성능 차이 또한 비교하고자 함**



## 👵 [AI-Hub의 낙상사고 위험동작 영상-센서 쌍 데이터를 다운로드](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71641)

* 'D:\Falldown\Dataset\Original_Dataset'를 다운로드 경로로 지정

* 01.원천데이터 -> SourceData (폴더 이름 변경)

* 02.라벨링데이터 -> LabelingData (폴더 이름 변경)



## 🧑‍🦼 5112개의 비낙상 데이터와 5096개의 낙상 데이터를 데이터셋으로 옮김 (낙상 데이터가 과도하게 많아 기존 15288개 중 1/3인 5096개만 사용하기로 함)

* 1 피실험자당 8개의 카메라 각도로 촬영됨

* 1개의 카메라 각도당 이미지는 10장, 비디오는 10초 촬영됨

    (10초/비디오 * 60 프레임/초 = 600 프레임)

* yolov8s 학습에는 데이터셋의 모든 이미지를 사용함

    (10208 * 10 = 102080 사진)

* GRU 학습에는 데이터셋의 1/5에 해당하는 비디오만 사용함

    (2024 동영상)

* 동영상은 1개당 600 프레임이며, 6프레임 별로 yolo bbox ratio 1개 및 mediapipe 관절 좌표값 22개를 추출함

    (100 프레임/동영상 * 23개/프레임 = 2300개/동영상)



## 🚨 기존 label에 기록된 낙상 중인 시점을 이용하여 100 프레임/동영상에 대해 각 프레임당 피실험자 상태(정상/낙상 중/낙상 후)를 classification함

* 2024개의 동영상(100 프레임/동영상)에 대해 yolo bbox, mediapipe 관절 좌표값을 탐지하여 JSON으로 추출함

    (팀원 GPU 이용)

* 우리는 이미 추출된 JSON에 맞춰 동영상 데이터셋을 생성하기로 함



### 🙋‍♂️ 구성원 : 조강희, 김덕휘, 민경원, 유혜민, 이현주