![image](https://github.com/user-attachments/assets/23db5374-eac4-4dbc-9d74-d7aaf0f48af5)# Project chorongE

저 시력자용 보조 비전
최종 목표는 잿슨나노 등을 이용하여 단독제품으로 구현

## High Level Design

* (프로젝트 아키텍쳐 기술, 전반적인 diagram 으로 설명을 권장)
* opencv yolo11, MediaPipe , tts
* AI models: hand detection (media-pipe), detection-train (yolo11), mono-depth (openvino) , tts (pyttsx3)

## Clone code

* (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)

```shell
git clone https://github.com/spotSide/projJewel.git
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)

```shell
python -m venv .openvino_env
source .openvino_env/bin/activate
cd proj
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/home/intel/openvino_env
source .openvino_env/bin/activate
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/home/intel/openvino_env
source .openvino_env/bin/activate

cd /home/intel/openvino/proj/test_sep
python3 test_main.py 
```

## Output

* (프로젝트 실행 화면 캡쳐)

![./result.jpg](./result.jpg)

## Appendix

* monodepth, mediaPipe, tts등을 사용하는 프로젝트
* 시각장애인용 솔루션을 제시하는 것이 목표
