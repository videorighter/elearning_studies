# 순환신경망(RNN)을 활용한 이러닝 학습자의 집중도 판별 연구
최근 코로나19로 인해 비대면 원격수업이 활성화 되면서 이러닝에 대한 관심이 증가하고 있다. 이러닝은 여러 가지 장점에도 불구하고 학습자와 교수자 간의 직접적인 교류가 부족하기 때문에 즉각적인 피드백을 받기가 힘들다는 단점이 있다. 이러한 단점을 보완하기 위해 다양한 방법론이 개발되었으나, 학습자의 태도를 정량적으로 측정하여 분석한 연구는 부족한 실정이다. 본 연구에서는 이러닝 학습자 영상에서 추출한 시선 및 얼굴 윤곽 데이터에 순환신경망 모델을 적용해 학습자의 집중도를 예측하는 방법론을 제시하고자 한다. 이러닝 학습자 92명을 촬영한 184개의 영상으로부터 OpenFace 2.0 툴킷을 이용해 프레임 별 데이터를 추출했고 5초 단위로 분할해 레이블링을 진행했다. 전처리한 데이터에 순환신경망(Recurrent neural network), 장단기메모리(Long-short term memory), 게이트순환유닛(Gated recurrent units) 모델을 적용하여 비교분석하였다. 본 연구는 교수자에게는 학습자 집중 여부에 따른 이러닝 컨텐츠 개발에 대한 가능성을 제시할 수 있고, 이러닝 학습자에게는 스스로 학습 태도를 관리할 수 있는 기회를 제공함으로써 교육 효과 상승을 기대할 수 있다는 점에서 의의가 있다. <br><br>

### How to use.
```
$ python3 data_preprocessing_keras_bi.py
$ python3 model_rnn_BI_BN_IN.py --repeats [INTEGER] --verbose [0|1|2] --epochs [INTEGER] --batch_size [INTEGER] --drop_out [FLOAT] --lr [FLOAT] --n_hidden [INTEGER] --decay [FLOAT] --device ['cpu'|'gpu]    
$ python3 model_lstm_BI_BN_IN.py --repeats [INTEGER] --verbose [0|1|2] --epochs [INTEGER] --batch_size [INTEGER] --drop_out [FLOAT] --lr [FLOAT] --n_hidden [INTEGER] --decay [FLOAT] --device ['cpu'|'gpu]
$ python3 model_gru_BI_BN_IN.py --repeats [INTEGER] --verbose [0|1|2] --epochs [INTEGER] --batch_size [INTEGER] --drop_out [FLOAT] --lr [FLOAT] --n_hidden [INTEGER] --decay [FLOAT] --device ['cpu'|'gpu]
```
### Code description
>data_preprocessing_keras_bi.py
>>전처리 코드
>>데이터 로드, Feature 추출 및 Label 부여
>>5초단위(150 frame) split
>>Train/Vaildation/Test dataset split and save

>>label_preprocess.py
>>>레이블 전처리 코드
>>>5초 단위 레이블을 dictionary 형태로 변형

>model_rnn_BI_BN_IN.py
>>Vanilla RNN 실행 코드

>model_lstm_BI_BN_IN.py
>>LSTM 실행 코드

>model_gru_BI_BN_IN.py
>>GRU 실행 코드

>cheack_target.py
>>target 분포 확인

>check_file_len.py
>>파일 길이 확인

### Dataset
[OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) 동영상으로부터 시선, 얼굴각도, 감정데이터를 추출하는 toolkit
<img width="1015" alt="스크린샷 2021-05-26 오후 1 46 35" src="https://user-images.githubusercontent.com/75473005/119604085-6e654800-be29-11eb-8dc6-f4148c067db1.png">

### Results
<img width="1016" alt="스크린샷 2021-05-26 오후 1 52 47" src="https://user-images.githubusercontent.com/75473005/119604283-c439f000-be29-11eb-8842-1dfe2cca2dff.png">
<img width="1015" alt="스크린샷 2021-05-26 오후 1 53 00" src="https://user-images.githubusercontent.com/75473005/119604290-c603b380-be29-11eb-92b8-f62fa9b5d531.png">
<img width="1014" alt="스크린샷 2021-05-26 오후 1 53 10" src="https://user-images.githubusercontent.com/75473005/119604297-c7cd7700-be29-11eb-8fa9-145672b878a6.png">
