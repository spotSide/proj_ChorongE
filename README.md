시각장애인 타겟의 보조 비전
최종 목표는 잿슨나노 등을 이용하여 단독제품으로 구현

## High Level Design

![image](https://github.com/spotSide/proj_ChorongE/blob/main/pic/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-12-19%2015-47-06.png)

## Clone code

* (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)

```shell
git clone https://github.com/spotSide/projJewel.git
```

## Prerequite


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


```shell
cd ~/home/intel/openvino_env
source .openvino_env/bin/activate

cd /home/intel/openvino/proj/test_sep
python3 test_main.py 
```

## Output

* (프로젝트 실행 화면 캡쳐)

![image](https://github.com/user-attachments/assets/23db5374-eac4-4dbc-9d74-d7aaf0f48af5)# Project chorongE

## Appendix

* monodepth, mediaPipe, tts등을 사용하는 프로젝트
* 시각장애인용 솔루션을 제시하는 것이 목표
