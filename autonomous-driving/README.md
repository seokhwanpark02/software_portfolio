# 카메라·LiDAR 융합 자율주행 시스템

1/10 스케일 차량(Xycar)에서 **카메라·LiDAR 인식 → 미터 단위 BEV → 상황별 경로 CNN → 경로 검사 → 조향·속도 제어**를 하나의 ROS 2 주행 소프트웨어로 연결한 프로젝트입니다.
국민대학교 자율주행 경진대회 본선에서 전체 코스를 완주했습니다(예선 130팀 중 본선 진출, 본선 22팀 중 8위).

| 기간 | 팀 규모 | 역할 |
|---|---|---|
| 2026.05 ~ 2026.08 | 5인 | 인지–제어 파이프라인 통합·검증 |

**직접 맡은 부분**
- 녹화 데이터를 다시 재생하는 Replay 환경과 파이프라인 Viewer 구축 → [3. 검증 도구](#3-검증-도구)
- CNN 경로가 제어로 넘어가기 전의 검사 기준 설계 → [2.5 경로 검사](#25-경로-검사-데이터로-정한-기울기곡률-상한)
- VESC 계측 중계 노드 작성, 출발 속도 상승률 결정, 하위 노드 블로킹 램프 제거 → [2.4 출발 구간 전류](#24-출발-구간-전류-vesc-계측으로-상승률-결정)

> **원본 저장소:** [KHUSLA-SANDI/KOOKMIN](https://github.com/KHUSLA-SANDI/KOOKMIN)
>
> 이 폴더는 위 원본 저장소의 `src/`에서 **인지·판단·제어 핵심 모듈과 테스트, 실행 설정**만 가져와 정리한 것입니다.
> 학습된 모델 가중치, 녹화 데이터, 차량 제조사 제공 드라이버는 포함하지 않았습니다.

▶ **본선 주행 영상:** [제9회 국민대학교 자율주행 경진대회 AI - 엣지챌린지 공식 영상 (4:24:32 ~ 4:27:32)](https://www.youtube.com/watch?v=CcfXS3UFL0A&t=15872s)

![architecture](docs/architecture.png)

## 목차

1. [시스템 구성](#1-시스템-구성)
2. [문제와 원인 → 해결 과정 → 결과](#2-문제와-원인--해결-과정--결과)
   - [2.1 네 좌표계를 거꾸로 잇는 BEV 변환](#21-네-좌표계를-거꾸로-잇는-bev-변환)
   - [2.2 Replay로 재현한 추론 지연과 CPU 경합](#22-replay로-재현한-추론-지연과-cpu-경합)
   - [2.3 LiDAR 채널의 학습–실차 불일치와 시간 동기화](#23-lidar-채널의-학습실차-불일치와-시간-동기화)
   - [2.4 출발 구간 전류: VESC 계측으로 상승률 결정](#24-출발-구간-전류-vesc-계측으로-상승률-결정)
   - [2.5 경로 검사: 데이터로 정한 기울기·곡률 상한](#25-경로-검사-데이터로-정한-기울기곡률-상한)
   - [2.6 추월: 결정론적 조향 블록과 테스트](#26-추월-결정론적-조향-블록과-테스트)
3. [검증 도구](#3-검증-도구)
4. [디렉터리 구조](#4-디렉터리-구조)
5. [빌드와 테스트](#5-빌드와-테스트)

---

## 1. 시스템 구성

| 단계 | 하는 일 | 주요 파일 |
|---|---|---|
| **인지** | YOLO segmentation으로 흰 차선·노란 중앙선·신호등·라바콘 구간 표식을 검출하고, 마스크를 차량 기준 미터 좌표 BEV로 변환 | `yolo_bev_node.py`, `bev_geometry.py` |
| **센서 융합** | LiDAR 점을 같은 BEV 격자에 투영하고, 카메라 시각과 가장 가까운 scan만 결합 | `lidar_geometry.py`, `lidar_compact.py`, `road_lidar_filter.py` |
| **판단** | 일반·지름길·추월·라바콘 4개 경로 CNN 중 현재 모드에 맞는 모델 하나만 실행, 신호·구간 미션 관리 | `cnn_path_node.py`, `cnn_modes.py`, `path_model.py`, `signal_mission.py` |
| **경로 검사** | CNN 경로의 유효점 수·길이·좌우 범위·기울기·곡률을 검사해 비정상 경로 차단 | `path_contract.py` |
| **제어** | Lookahead 기반 조향, IIR smoothing, 속도 slew ramp, 결정론적 추월 블록. 모터 명령은 이 노드 하나만 발행 | `simple_motion_node.py`, `hardcoded_overtake.py`, `overtake_block.py` |
| **차량 인터페이스** | VESC 전압·전류·고장 코드를 ROS 2 쪽으로 중계 | `vehicle_interface/vesc_relay.py` |
| **검증 도구** | 센서 녹화·원래 시간 간격 재생·파이프라인 시각화 | `replay_recorder_node.py`, `replay_source_node.py`, `live_pipeline_viewer_node.py` |

설계 원칙은 다섯 가지였습니다.

- **최신성 우선:** 모든 프레임을 처리하는 것보다 지금 장면에 반응하는 것을 우선합니다.
- **좌표계 통일:** 카메라와 LiDAR를 하나의 미터 단위 차량 좌표계로 맞춥니다.
- **전처리 계약:** 학습할 때와 달릴 때 같은 전처리를 거칩니다.
- **시간적 확인:** 한 프레임의 인지 결과보다 여러 프레임에 걸쳐 확인된 사건을 믿습니다.
- **권한 분리:** 경로를 만드는 노드와 모터 명령을 내는 노드를 분리합니다.

---

## 2. 문제와 원인 → 해결 과정 → 결과

### 2.1 네 좌표계를 거꾸로 잇는 BEV 변환

**문제와 원인**
- 카메라 마스크가 BEV에 놓이려면 네 개의 좌표계를 거쳐야 합니다: YOLO letterbox 좌표 → 렌즈 왜곡이 남은 원본 픽셀 → 왜곡을 지운 픽셀 → 차량 기준 미터 좌표.
- 기존에는 대응점을 찍어 RANSAC으로 구한 호모그래피를 썼는데, 이 H는 왜곡을 지운 원본 해상도 기준이고 YOLO 마스크는 letterbox가 적용된 저해상도 좌표여서 두 좌표계가 섞여 있었습니다. 대응점이 어디에 몰렸느냐에 따라 거리와 좌우 오차도 달라졌습니다.
- 변환 뒤 후처리에서도 정보가 사라졌습니다. 전방 5 cm 간격마다 차선 픽셀의 횡방향 중앙값 한 점만 남기다 보니, 갈림길처럼 같은 거리에 차선이 둘 이상 있으면 하나로 뭉개졌고, 점선 중앙선은 몇 조각만 남아 경로 곡선이 쉽게 휘었습니다.

**해결 과정**
- **물리 모델 기반 호모그래피:** 대응점 대신 카메라 자체의 파라미터로 지면 평면 변환을 계산했습니다.
  - 내부 파라미터 `K`, 왜곡 계수 `D`, 왜곡 보정 영상용 `new_K`를 따로 관리합니다.
  - 외부 파라미터는 장착 높이(약 0.20 m), 피치(약 7.24°), 요·롤, LiDAR와의 전후 간격으로 정합니다.
- **역방향 remap 테이블:** 호모그래피·렌즈 왜곡·letterbox 변환을 하나로 결합해, 각 BEV 셀이 참조할 원본 마스크 좌표를 미리 계산해 두었습니다.
  1. 셀의 미터 좌표를 `H⁻¹`로 왜곡 보정 영상 픽셀에 투영
  2. `K`, `D`, `new_K`로 렌즈 왜곡을 다시 입혀 원본 픽셀로 복원
  3. letterbox scale·padding을 적용해 YOLO 마스크 좌표로 변환
  4. `cv2.remap()`으로 샘플링하고, 화면 밖 영역과 차량 보닛은 마스킹
- 테이블은 첫 유효 마스크가 들어올 때 만들고 입력 해상도가 같으면 재사용합니다. 셀 안은 3배로 supersampling해 얇은 차선이 빠지지 않게 했습니다.
- **마스크를 그대로 쓰는 표현:** 중앙값으로 줄이던 단계를 없애고, 흰선·노란선·LiDAR를 셀당 2.5 cm인 3×128×120 BEV 채널로 그대로 넣었습니다. 경로 CNN은 전방 0.3~3.0 m 구간의 고정된 28개 지점마다 횡방향 위치와 유효도를 출력합니다.

**결과**
- 주행 시퀀스 단위로 따로 떼어 둔 테스트 321장에서 **경로 예측 MAE 8.045 cm**
- 트레이드오프: 마스크 표현으로 바꾸면서 학습용 정답 경로를 처음부터 만들어야 했습니다. 경로 후보를 자동으로 만든 뒤 사람이 확인해 고쳤고, 비슷한 연속 프레임이 학습·평가 양쪽에 들어가지 않도록 데이터를 주행 시퀀스 단위로 갈랐습니다.

관련 코드: [`bev_geometry.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/bev_geometry.py) (`BevMaskRemapper`, `_ground_to_undistorted`, `_redistort`), [`config/camera.yaml`](track_drive_cnn_gpt/config/camera.yaml), [`path_contract.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/path_contract.py)

### 2.2 Replay로 재현한 추론 지연과 CPU 경합

**문제와 원인**
- 검출 모델만 따로 돌리면 프레임당 50 ms 안팎이었지만, 전체 파이프라인을 올린 채 달리면 평균 181 ms 정도까지 늘어났습니다.
- 실차에서는 같은 장면을 반복할 수 없어 원인을 좁히기 어려웠습니다. 녹화한 센서 데이터를 원래 시간 간격대로 다시 내보내는 Replay 환경에서 CPU 부하를 올려 같은 지연을 만들고, 프로세스별 사용률을 비교했습니다.
- 브리지의 영상 전송·변환 부하가 주요 원인이었습니다. 모든 토픽을 중계하던 ROS 1–2 브리지가 1080p 원본과 압축 영상 여러 종류까지 변환하느라 추론과 CPU를 나눠 쓰고 있었습니다.
- 여기에 카메라 입력이 추론보다 빠르면 처리하지 못한 프레임이 큐에 쌓여, 검출이 정확해도 이미 지나온 장면을 기준으로 조향하게 되는 문제가 더해졌습니다.

**해결 과정**
- **전송 정리:** 브리지는 필요한 토픽만 연결하도록 바꾸고, 같은 장치 안의 영상 전송은 공유 메모리(SHM)로 넘겼습니다.
- **큐 적체 차단:** 처리 대기 중인 프레임은 새 프레임이 오면 덮어쓰는 Latest-Only 버퍼를 두고, 센서 QoS를 `KEEP_LAST`, depth 1로 고정했습니다.
- **추론 비용 절감:**
  - YOLO와 신호등 분류기를 OpenVINO FP32, static batch 1로 변환하고 시작할 때 warm-up을 돌립니다.
  - 신호등 분류기는 신호등 박스가 잡힌 프레임에서만 crop에 대해 실행합니다.
  - 경로 CNN 네 개는 미리 올려 두되 현재 모드의 모델 하나만 추론합니다.
- **배포 안전장치:** 모델 파일의 SHA-256, 입력 크기, 클래스, precision, batch를 시작 시점에 검사해 맞지 않으면 노드를 띄우지 않습니다(fail-closed).

**결과**
- 실주행 평균 181 ms는 CPU 경합을 관찰한 값이고, 아래 p50 비교는 같은 녹화 프레임으로 따로 측정한 백엔드 벤치마크입니다.
- 같은 녹화 프레임 기준 **YOLO 추론 p50: 91.76 ms(PyTorch) → 51.03 ms(OpenVINO)** (차량 CPU, 입력 384×640)
- 경로 발행률 11.948 Hz
- 트레이드오프: 밀린 중간 프레임은 버려집니다. 주행 제어에서는 모든 프레임을 빠짐없이 처리하는 것보다 가장 최근 장면에 반응하는 것이 더 중요하다고 판단했습니다.

관련 코드: [`yolo_bev_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/yolo_bev_node.py) (`_LatestOnlyBuffer`, `_sensor_qos_depth_one`), [`tools/benchmark_yolo_backends_gpt.py`](track_drive_cnn_gpt/tools/benchmark_yolo_backends_gpt.py), [`tools/export_openvino_gpt.py`](track_drive_cnn_gpt/tools/export_openvino_gpt.py), [`replay_source_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/replay_source_node.py)

### 2.3 LiDAR 채널의 학습–실차 불일치와 시간 동기화

**문제와 원인**
- 경로 CNN의 세 번째 입력 채널인 LiDAR가 실차에서는 도움이 되기는커녕 경로 예측을 흔들었습니다. 채널을 켠 경우와 끈 경우를 같은 장면에서 비교해 원인이 이 채널에 있다는 것을 확인했습니다.
- 들여다보니 채널에 담긴 내용 자체가 학습 때와 달랐습니다. 학습용 합성 장애물이 차지하는 BEV 셀은 27~67개였는데, 실차 scan은 벽·차체·사람까지 잡혀 94~498개 셀을 채웠습니다(장애물 점유 셀 수, 비교 샘플 기준).
- 카메라와 LiDAR는 수신 시각도 서로 달라, 시간을 맞추지 않으면 다른 순간의 두 센서 정보가 한 BEV에 겹칠 수 있었습니다.

**해결 과정**
- **채널 의미 맞추기:** 합성 장애물 대신 대회 트랙에서 직접 기록한 scan으로 학습셋을 다시 만들었고, 데이터셋을 만들 때와 차량에서 달릴 때 같은 LiDAR 전처리 규칙을 적용했습니다.
  - 0.05~8.0 m 범위만 쓰고 거리 보정 배율을 적용
  - 차체 반사(self-return) 영역 제거
  - 각 점을 3×3 셀로 넓혀 학습 데이터와 같은 형태로 표시
  - 일반·지름길·추월 모델은 흰 차선 바깥 점을 지우고, 라바콘 모델만 바깥 콘을 남김(콘 자체가 경로 경계이므로)
- **시간 동기화:** LiDAR scan을 1.0 s 동안 보관하고, BEV 시각과 0.18 s 이내로 가장 가까운 scan만 결합합니다. 0.35 s보다 오래된 BEV는 쓰지 않습니다. 잘못된 scan이 들어와도 빈 scan으로 메우지 않아, "장애물 없음"과 "데이터 없음"이 섞이지 않게 했습니다.
- **용도별 필터 분리:** CNN 입력용과 추월 판정용 LiDAR 필터를 따로 두었습니다. 추월 판정은 흰 경계에서 10 cm 이상 안쪽 점만 인정하는 등 더 보수적입니다.

**결과**
- 학습·추론 LiDAR 전처리 통일
- CNN 입력용과 추월 판정용 필터 분리
- 트레이드오프: 도로 밖 점을 지우면 주변 정보가 줄어듭니다. 하지만 일반 주행에서는 그 점들이 경로를 흔드는 잡음이었습니다.

관련 코드: [`road_lidar_filter.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/road_lidar_filter.py), [`lidar_geometry.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/lidar_geometry.py), [`lidar_compact.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/lidar_compact.py)

### 2.4 출발 구간 전류: VESC 계측으로 상승률 결정

**문제와 원인**
- 정지 상태에서 목표 속도를 주면 차량이 출발 직후 멈칫거리며 매끄럽게 가속하지 못했습니다.
- 모터 컨트롤러(VESC)의 고장 코드에는 아무 기록이 남지 않아, 코드 값만으로는 전기적 문제인지 소프트웨어 문제인지 판단할 근거가 없었습니다.
- 확인해 보니 원인은 두 가지가 겹쳐 있었습니다. 속도 명령이 급하게 오르면서 모터 전류가 상한 가까이 치솟았고, 하위 모터 노드가 자체 속도 램프를 도는 약 0.77 s 동안 새 명령을 처리하지 못했습니다.

**해결 과정**
- 먼저 계측 경로를 만들었습니다. VESC 상태 메시지(`vesc_msgs`)는 ros1_bridge를 통과하지 못하므로, ROS 1 쪽에서 전압·모터 전류·ERPM·duty·고장 코드를 `Float32MultiArray`로 바꿔 ROS 2로 넘기는 중계 노드를 작성했습니다. 상위 명령이 끊기면 정지 명령을 대신 내보내는 데드맨 가드도 옵션으로 넣었습니다.
- 이 경로로 20 Hz 제어 루프의 속도 상승률을 바꿔 가며 지상에서 전류를 비교했습니다.

| 틱당 상승률 | 출발 최대 전류 | 최저 전압 | VESC fault | 속도 명령 0→12 |
|---|---|---|---|---|
| +0.5 | 15.34 A (상한 근처) | - | - | 약 1.2 s |
| **+0.25** | **12.62 A** | 8.3 V | 0회 | 약 2.4 s |

- 가감속 제한은 `simple_motion_node`의 20 Hz 루프 한 곳에서만 계산하고, 하위 노드에서 명령을 붙잡던 램프는 없앴습니다.
- 가속과 감속을 다르게 다뤘습니다. 가속은 틱당 +0.25로 천천히 올리고, 감속은 틱당 2.0으로 빠르게 내리며, 정지 명령은 램프 없이 같은 틱에 0으로 보냅니다.

**결과**
- **출발 최대 전류 15.34 A → 12.62 A**, VESC fault 0회
- 트레이드오프: 목표 속도까지 걸리는 시간은 1.2 s에서 2.4 s로 늘었습니다. 출발 때마다 1초를 아끼는 것보다 전류 여유를 두고 출발을 반복하지 않는 쪽이 완주에 유리하다고 봤습니다.

관련 코드: [`vehicle_interface/vesc_relay.py`](vehicle_interface/vesc_relay.py), [`simple_motion_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/simple_motion_node.py), [`config/simple_motion.yaml`](track_drive_cnn_gpt/config/simple_motion.yaml) (`slew_speed_up_per_tick`, `slew_speed_down_per_tick`)

### 2.5 경로 검사: 데이터로 정한 기울기·곡률 상한

**문제와 원인**
- CNN 출력을 곧바로 조향에 쓰면, 값이 튀거나 급하게 꺾인 경로 한 프레임만으로도 차량이 엉뚱한 방향으로 핸들을 돌릴 수 있었습니다.

**해결 과정**
- 경로를 제어로 넘기기 전에 아래 조건을 검사하고, 하나라도 어긋나면 그 프레임의 경로를 버리고 직전 정상 경로를 유지합니다.
  - 유효점 6개 이상
  - 전방 길이 0.5 m 이상
  - 좌우 범위 ±1.5 m
  - 기울기·곡률 상한
- 상한을 감으로 정하면 둘 중 하나가 깨집니다. 너무 엄격하면 일반 차선에서 크게 갈라지는 지름길 경로가 버려지고, 너무 느슨하면 이상 경로가 통과합니다.

**결과**
- 학습용 경로 8,391개 중 지름길 경로의 기울기·곡률 99백분위수(3.078, 11.187)를 기준으로 상한을 **3.1과 12.0**으로 정했습니다.
- 지름길 경로의 기울기·곡률 분포를 참고해, 급격한 분기는 허용하면서 극단적인 출력은 제한하도록 했습니다.

관련 코드: [`path_contract.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/path_contract.py), 테스트 [`test_path_contract.py`](track_drive_cnn_gpt/test/test_path_contract.py)

### 2.6 추월: 결정론적 조향 블록과 테스트

**문제와 원인**
- 한 프레임의 오검출로 주행 모드가 바뀌는 일이 있었습니다.
- 추월 전용 CNN은 장면마다 출력 경로가 달라, 같은 상황에서 같은 동작을 하는지 확인하기 어려웠습니다.

**해결 과정**
- **오검출 방지:** 같은 쪽 장애물이 2프레임 연속 확인돼야 추월을 시작하고, 5프레임 동안 장애물이 보이지 않아야 다시 감지합니다.
- **결정론적 추월:** LiDAR로 장애물이 왼쪽인지 오른쪽인지 판정한 뒤 `SHIFT → COUNTER → PASS` 3단계 조향 블록을 실행합니다.
- **모호한 입력 거부:** 장애물이 중앙 ±0.10 m 안에 있거나 양쪽에 동시에 있으면 판정하지 않습니다.
- **명령 충돌 방지:** 블록이 실행되는 동안에도 모터 명령은 `simple_motion_node` 하나만 발행해, 경로 추종과 추월 명령이 부딪치지 않게 했습니다.

**결과**
- 좌우 판정, 단계별 틱 수, 모호한 입력 거부, 중복 이벤트·재진입 방지를 포함한 집중 테스트 **36 passed, 2 skipped** (상태 전이·로직 테스트 기준)
- 최종 추월 구성은 실차 모터 시험 없이 위 로직 테스트로 검증했습니다.
- 트레이드오프: 장애물 위치에 맞춰 경로를 바꾸는 유연성은 포기했습니다. 대신 단계별 조향각과 틱 수가 고정돼 기대 결과를 미리 정의하고 테스트할 수 있습니다.

관련 코드: [`hardcoded_overtake.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/hardcoded_overtake.py), [`overtake_block.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/overtake_block.py), 테스트 [`test_hardcoded_overtake.py`](track_drive_cnn_gpt/test/test_hardcoded_overtake.py)

---

## 3. 검증 도구

2.2의 지연 분석처럼, 실차에서 재현하기 어려운 문제는 아래 도구로 같은 입력을 반복해 확인했습니다.

- **Recorder:** 실제 카메라·LiDAR 토픽을 기록합니다.
- **Replay Source:** 기록한 데이터를 원래 시간 간격과 같은 QoS로 다시 발행합니다. 실제 센서 노드와 replay 노드가 같은 토픽을 동시에 발행하면 실행을 거부합니다.
- **Viewer:** 카메라, BEV, CNN에 실제로 들어간 LiDAR 채널, 선택 경로, 모드, 조향 명령을 한 화면에 보여 줍니다. 제어 토픽을 발행하지 않는 read-only 구조라 차량 명령에 영향을 주지 않습니다.
- **진단 로그:** YOLO 추론 시간, 전체 지연, CNN 성공·실패 사유를 기록합니다.

덕분에 "모델 결과가 이상하다"에서 멈추지 않고, 카메라 검출·좌표 변환·LiDAR 필터·모드 전환·경로 출력 중 어느 단계에서 문제가 생겼는지 분리해서 볼 수 있었습니다.

관련 코드: [`replay_recorder_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/replay_recorder_node.py), [`replay_source_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/replay_source_node.py), [`live_pipeline_viewer_node.py`](track_drive_cnn_gpt/track_drive_cnn_gpt/live_pipeline_viewer_node.py), [`launch/replay_perception_cnn.launch.py`](track_drive_cnn_gpt/launch/replay_perception_cnn.launch.py)

---

## 4. 디렉터리 구조

```
autonomous-driving/
├── docs/architecture.png
├── vehicle_interface/
│   └── vesc_relay.py              # ROS 1(noetic) 컨테이너에서 실행하는 VESC 텔레메트리 중계
└── track_drive_cnn_gpt/           # ROS 2 (ament_python) 패키지
    ├── track_drive_cnn_gpt/       # 노드와 라이브러리
    ├── config/                    # 카메라 보정, 인지·제어 파라미터, DDS 설정
    ├── launch/                    # 주행·인지 단독·replay·viewer 실행 구성
    ├── test/                      # pytest 단위 테스트
    └── tools/                     # OpenVINO 변환·백엔드 벤치마크·점검 스크립트
```

## 5. 빌드와 테스트

**차량 실행 환경:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, Ultralytics 8.3.91, OpenVINO 2026.3.0 (`requirements_models_gpt.txt` 참고). VESC 중계 노드는 ROS 1 Noetic 컨테이너에서 실행하고 ros1_bridge로 연결했습니다.

**ROS 없이 단위 테스트만 실행** (Python 3.10 이상)

```bash
cd autonomous-driving/track_drive_cnn_gpt
pip install pytest pyyaml numpy opencv-python-headless
python3 -m pytest test/ \
  --ignore=test/test_replay_contract.py \
  --ignore=test/test_yolo_coordinate_preview.py
# ROS·PyTorch 없는 환경 기준: 191 passed, 12 skipped
```

- `test_replay_contract.py`, `test_yolo_coordinate_preview.py`는 `rclpy`를 import하므로 ROS 2 환경에서만 실행됩니다.
- 건너뛰는 12개는 PyTorch·`sensor_msgs`가 필요하거나, 차량에만 있는 파일을 대조하는 테스트입니다.

**ROS 2 환경에서 빌드**

```bash
# ROS 2 워크스페이스의 src/ 아래에 패키지를 두고
colcon build --packages-select track_drive_cnn_gpt
source install/setup.bash
python3 -m pytest src/track_drive_cnn_gpt/test/
```

**포함하지 않은 것:** 학습된 모델 가중치(`.pt`, OpenVINO IR), 녹화 데이터, 차량 제조사 제공 드라이버. `config/*.yaml`의 모델 경로는 대회 차량 기준입니다.

**수치의 측정 조건**

| 수치 | 조건 |
|---|---|
| 경로 MAE 8.045 cm | 주행 시퀀스 단위로 분리한 테스트 321장, 전방 28지점 횡방향 오차 평균 |
| YOLO p50 91.76 → 51.03 ms | 같은 녹화 프레임, 차량 CPU, 입력 384×640, PyTorch vs OpenVINO FP32 batch 1 |
| 출발 전류 15.34 → 12.62 A | 20 Hz 제어, 지상 실차 측정, 속도 명령 0→12 |
| 추월 테스트 36 passed | `hardcoded_overtake` 상태 전이·로직 집중 테스트 |
