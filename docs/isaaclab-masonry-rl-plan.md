# Isaac Lab 조적(Masonry / Bricklaying) 강화학습 프로젝트 계획서

> 상태: **v0.5 (자세 해석 반영)** · 대상: Isaac Lab 2.x (Isaac Sim 4.5+) · 목표: 시뮬레이션 데모까지

## 확정된 프로젝트 조건

| 항목 | 확정 내용 |
|---|---|
| 로봇 | **Boston Dynamics Atlas** (이족 휴머노이드, 양팔) — 공개 자산 조사 결과 **DRC Atlas** 사용, 전기 Atlas는 자산 비공개 (§1) |
| 이동 | **제자리 고정** — 보행 없음. 단, **균형 유지 + 스쿼트/굴신은 필수** |
| 작업 | **바닥에서부터 소형 벽 축조** (4장 × 3단, 약 790 × 211 mm) |
| 폴백 | 바닥 시공 미수렴 시 **기존 3단 위에서 4~6단 시공**(`base_courses=3`)으로 데모 확정 (§11.1) |
| 몰탈 | **포함** — 도포(압출) → 압착 착좌 → 경화 전 과정 |
| 최종 산출물 | **시뮬레이션 데모** (학습 정책 + 렌더링 영상), 실기 이식 없음 |
| 학습 하드웨어 | **NVIDIA RTX A6000 48 GB** 단일 GPU |

---

## 0. 한 줄 정의

제자리에 선 **Atlas가 균형을 유지하며 굴신·스쿼트로 자세를 바꿔 가며, 바닥의 팔레트에서 벽돌을 집어 → 몰탈 베드를 압출 도포 → 목표 슬롯에 정렬·압착 착좌 → 놓기**를 반복해 3단 소형 벽을 완성하는 정책을, Isaac Lab GPU 병렬 시뮬레이션에서 PPO로 학습하고 데모 영상으로 산출한다.

**최종 성공 기준**

| 지표 | 목표 |
|---|---|
| 벽 완성률 (12장 중 배치 성공) | ≥ 70% |
| 장당 배치 성공률 (위치 ≤ 5 mm, yaw ≤ 2°) | ≥ 85% |
| 줄눈 두께 8~12 mm 유지율 | ≥ 80% |
| 기 배치 벽돌 붕괴율 | ≤ 5% |
| **전도(넘어짐) 발생률** | **≤ 1%** |
| 데모 | 1~4 env 고품질 렌더 영상 (몰탈 압출·스퀴즈아웃 시각화 포함) |

---

## 1. Atlas 자산 조사 결과 (2026-07 실측)

E-Atlas(전기 Atlas)에 대해 실제로 조사한 결과를 정리합니다. **결론부터: 사례는 풍부하게 존재하지만, 공개 자산은 존재하지 않습니다.** 이 둘은 구분해야 합니다.

### 1.1 확인된 사실 — E-Atlas + Isaac Lab 사례는 실재한다

| 사실 | 출처 |
|---|---|
| Boston Dynamics와 연구 파트너들이 **Isaac Lab에서 전기 Atlas의 학습 기반 조작·보행 정책을 개발 중** | [Boston Dynamics × NVIDIA 협력 발표](https://bostondynamics.com/news/boston-dynamics-expands-collaboration-with-nvidia/) |
| NVIDIA GTC 키노트에서 **Atlas 디지털 트윈**이 Hyundai 공장 작업을 시뮬레이션 학습하는 장면 공개 | [The Robot Report](https://www.therobotreport.com/boston-dynamics-plans-to-use-nvidias-isaac-gr00t-to-build-ai-capabilities-for-atlas/) |
| Atlas가 **약 45 kg(100 lb) 산업용 화물을 전신(whole-body) RL + 대규모 시뮬레이션으로 학습해 들어올림**, 시뮬 정책을 실기에 **zero-shot 전이** | [Interesting Engineering](https://interestingengineering.com/ai-robotics/boston-dynamics-atlas-humanoid-heavy-lifting-simulation) |
| CES 2026에서 **양산형 Atlas 공개, 56 DoF**, Google DeepMind AI 탑재 | [Humanoids Daily](https://www.humanoidsdaily.com/news/the-alien-in-the-factory-boston-dynamics-launches-production-ready-atlas-at-ces-2026) |

**이 사례들은 우리 계획에 직접적인 설계 근거가 됩니다.** 특히 "무거운 물체를 전신 RL로 들어올리고 zero-shot 전이"는 조적 태스크(무거운 물체 + 이족 균형 + 정밀 배치)와 구조가 거의 같습니다. §12에 설계 시사점을 정리했습니다.

### 1.2 확인된 사실 — 그러나 공개 자산은 없다

두 곳을 직접 확인했습니다.

| 확인 대상 | 결과 |
|---|---|
| **Isaac Lab `isaaclab_assets/robots/`** ([소스](https://github.com/isaac-sim/IsaacLab)) | 등록된 로봇: Agibot, Agility, Allegro, Ant, ANYmal, Cassie, Fourier, Franka, Galbot, Humanoid, Humanoid_28, Kinova, Kuka-Allegro, Quadcopter, Ridgeback-Franka, Sawyer, Shadow Hand, **Spot**, Unitree, Universal Robots. → **Boston Dynamics는 Spot(사족)뿐, Atlas 없음** |
| **`robot_descriptions.py`** (90+ 로봇, 휴머노이드 40종) ([소스](https://github.com/robot-descriptions/robot_descriptions.py)) | Atlas 항목은 딱 2개 — `atlas_drc_description` (**Atlas DRC v3**, URDF, Drake 유래), `atlas_v4_description` (**Atlas v4**, URDF, roboschool 유래). → **둘 다 DRC 시절 유압 Atlas. 전기 Atlas 없음** |

즉 GTC에서 본 Atlas 시뮬레이션은 **Boston Dynamics 사내 자산**이며, 외부에 배포되지 않았습니다. 실기 자체도 2026년 물량이 Hyundai RMAC과 Google DeepMind에 전량 배정되어 외부 판매가 없는 상태라, 자산 공개 유인도 낮습니다.

### 1.3 그래서 어떻게 할 것인가 — 3안

**A안 (권장) — DRC Atlas URDF 즉시 확보**

가장 확실하고 빠릅니다. `pip install robot_descriptions` 한 줄로 URDF 경로가 나옵니다.

```python
from robot_descriptions import atlas_drc_description   # Atlas DRC v3 (Drake 유래)
from robot_descriptions import atlas_v4_description    # Atlas v4 (roboschool 유래)
# → .URDF_PATH 로 Isaac Lab URDF Importer에 바로 투입
```

- 스펙: 약 30 DoF (양팔 6+6, 양다리 6+6, 허리 3, 목 1), 신장 약 1.88 m, **질량 약 175 kg**
- 장점: **진짜 Boston Dynamics Atlas 형상**, 즉시 접근 가능, 패키지화되어 있어 P0 리스크가 크게 낮아짐
- 단점 및 대응:
  - **손이 없음** → 손목에 평행 그리퍼(Robotiq 2F-140 등) 부착 (자산 작업 1건)
  - 유압 구동계 → `ImplicitActuatorCfg`로 근사, 게인 재튜닝
  - 175 kg 중량체 → 벽돌 2.4 kg은 무시할 수준이나 **스쿼트/균형 동역학이 무겁고 느림**. 다만 관성이 커서 외란에 둔감한 이점도 있음
  - 구형 URDF → 관절 리밋·관성 텐서 검수, 충돌 메시 convex hull 단순화 필수
  - 각 리포지토리 라이선스 확인 (P0 체크리스트)

**B안 — 전기 Atlas 근사 자체 제작**: 공개 스펙(56 DoF, 신장 약 1.5 m, 질량 약 89 kg)과 영상 기반 모델링. 정확도 낮고 비용 큼. **권장하지 않음.**

**C안 (폴백) — 대체 휴머노이드**: Isaac Lab 기본 제공 Unitree G1(29-DoF) 등으로 파이프라인을 완성. 전기 Atlas 자산이 향후 공개되면 교체.

> **핵심 엔지니어링 판단: 환경을 처음부터 "로봇 독립적(robot-agnostic)"으로 설계합니다.**
> `masonry_env_cfg.py`가 로봇을 하드코딩하지 않고 `RobotPresetCfg`(관절 이름 그룹, EE 프레임, 리치, 기본 자세, 액추에이터 게인)를 주입받게 만들면 **Atlas ↔ G1 교체가 config 한 줄**이 됩니다. 비용이 거의 안 드는 보험이므로 무조건 넣습니다. 이 설계 하나로 "전기 Atlas가 나중에 공개되면 갈아탄다"도 자연히 성립합니다.

**진행 방침: A안(DRC Atlas) 1순위, C안 상시 가능하도록 로봇 독립 설계.** 이하 문서는 A안 기준 수치입니다.

---

## 2. Atlas가 Spot보다 어려운 이유 — 계획을 지배하는 4가지

이족 휴머노이드로 바뀌면서 난이도의 성격 자체가 달라집니다. 사족(Spot)에서는 "흔들림"이었던 것이 이족에서는 **"전도"** 가 됩니다.

**(1) 지지 다각형이 발 2개뿐이다 — 넘어질 수 있다.**
Spot은 사족이라 지지면이 넓어 팔 반력에 흔들리는 정도지만, Atlas는 **약 0.3 × 0.25 m 지지 다각형** 위에 서 있습니다. 2.4 kg 벽돌을 전방 0.7 m로 뻗고 30~80 N 압착력을 가하면 CoM이 지지 다각형 밖으로 나가 **넘어집니다.** → 균형은 보상 항목이 아니라 **별도로 먼저 학습해야 하는 저수준 정책**입니다.

**(2) 바닥 작업 = 깊은 스쿼트 = 전신 동작.**
"바닥부터"라는 조건 때문에 1단 배치와 팔레트 픽업이 **모두 바닥 높이**에서 일어납니다. Spot은 몸을 낮추면 그만이지만, Atlas는 **무거운 팔을 뻗은 채 무릎·고관절·허리를 굽혀 스쿼트하면서 균형을 유지**해야 합니다. 이 프로젝트에서 가장 어려운 단일 구간입니다.

> **이 문제를 커리큘럼으로 뒤집습니다: 학습을 바닥이 아니라 높은 곳에서 시작합니다.**
> 벽은 아래에서 위로 쌓지만, **학습 순서는 그럴 필요가 없습니다.** 이를 위해 **`base_courses`(사전 축조 단수)** 라는 단일 파라미터를 도입합니다.
>
> ```
> base_courses = N  →  이미 N단이 쌓여 있는 벽 위에서 (N+1)단부터 시공
>                      사전 축조분은 kinematic(경화 완료)으로 스폰 — 붕괴 위험 없음, 연산 비용 낮음
>                      작업면 높이 = 10 + 67·N [mm]
> ```
>
> | `base_courses` | 작업면 높이 | Atlas 자세 |
> |---|---|---|
> | 12 | 약 0.82 m | 선 자세 — 균형 부담 최소 |
> | 6 | 약 0.41 m | 중간 굴신 |
> | 3 | 약 0.21 m | 얕은 스쿼트 |
> | 0 | 약 0.01 m | **깊은 스쿼트 — 최난이도** |
>
> 커리큘럼은 `base_courses`를 **12 → 6 → 3 → 0으로 내리는 것**만으로 균형 난이도를 연속 조절합니다(L5~L6). 벽을 아래부터 쌓는 순서는 학습이 끝난 정책을 실행할 때만 지키면 됩니다. **이 계획서에서 가장 중요한 커리큘럼 설계이며, 동시에 §11의 폴백 장치이기도 합니다.**

**(3) 양팔이 있다 — 기회이자 비용.**
Atlas는 양팔이라 **한 팔은 몰탈 노즐, 다른 팔은 벽돌**이라는 실제 조적공에 가까운 분업이 가능합니다. Spot(단완)보다 태스크가 자연스럽고 데모 임팩트도 큽니다. 다만 액션 차원이 2배가 되므로 **단완 → 양팔 순으로 단계 투입**합니다(P6).

**(4) DoF가 30개다.**
관측·액션 차원이 커지고 환경당 연산량이 늘어 **병렬 환경 수가 줄고 학습 시간이 길어집니다**(§8). 골반 고정 단계에서 다리를 비활성화해 초반 학습 비용을 낮추는 것이 중요합니다.

---

## 3. 태스크 분해 — 3층 구조

조적은 long-horizon + contact-rich + 희소 보상 조합이고, 여기에 이족 균형까지 얹혔습니다. 통짜 학습은 확실히 실패합니다. **3층으로 분리합니다.**

```
[3층] 슬롯 플래너 (학습 아님, 결정론적)
      단/열 순서 → 목표 슬롯 pose 생성. wall_planner.py

[2층] 조작 정책 (RL, 본 프로젝트의 산출물)
      "지정 슬롯 1곳에 몰탈 도포 후 벽돌 1장 안전 착좌"
      출력: 양팔 task-space delta, 그리퍼, 압출률, + 골반/CoM 명령

[1층] 균형 정책 (RL, 사전 학습 후 동결)
      "제자리에서 외란에 저항하며 지정 골반 pose 유지"
      입력: 2층의 골반 명령(높이/피치/전후·좌우 shift)
      출력: 다리 12 DoF 관절 목표
```

**1층을 먼저 따로 학습해 동결한 뒤 2층을 학습**합니다(계층적 RL). 2층이 균형까지 동시에 배우게 하면 탐색 공간이 폭발합니다. 1층 학습은 벽돌 없이 돌리므로 빠릅니다(§8.4, 약 1시간).

### 벽 사양 (Atlas 리치 기준 확정)

```
벽돌   : 표준형 190 × 90 × 57 mm, 질량 2.4 kg (±10% 랜덤화)
줄눈   : 목표 10 mm (허용 8~12 mm)
1단    : 4장 → 4×190 + 3×10 = 790 mm
단높이 : 57 + 10 = 67 mm → 3단 상단 = 201 mm, 4단 작업면 = 211 mm
벽돌수 : 12장 + 어긋쌓기용 반절 4장
배치   : 벽을 Atlas **정면에 가로로** 배치 (기준면 전방 0.45 m, 좌우 ±0.395 m),
         팔레트는 우전방 (0.30, −0.55). 최원거리 슬롯 0.57 m < 리치 0.80 m
         ※ 벽을 전방으로 뻗게 놓으면 far end가 1.44 m로 리치를 크게 초과한다.
           제자리 고정 로봇이라 가로 배치가 강제된다 (scripts/preview_wall.py로 검증)

base_courses : 사전 축조 단수 (커리큘럼·폴백 파라미터, §2·§5.5·§11)
  = 0  → 바닥부터 3단 자력 시공  [최종 목표]
  = 3  → 기존 3단 위에 4~6단 시공 [확정 폴백]
  = 12 → 기존 12단 위에 13~15단 시공 [학습 초기]
  사전 축조분은 kinematic 강체로 스폰(경화 완료 상태) — 붕괴 위험 없음, 연산 부담 최소
```

> 3단(211 mm)은 Atlas 입장에서 전 구간이 스쿼트 영역입니다. 학습이 잘 풀리면 **5단(345 mm)으로 확장**해 서기→굽히기 전 범위를 보여주는 쪽이 데모 임팩트가 큽니다. 스코프 관리를 위해 기본은 3단, 확장은 P7 선택 항목입니다.

---

## 4. 몰탈 모델링 — 2트랙 전략

**학습용과 데모용 몰탈 모델을 분리합니다.** PBD 입자 유체를 수백~수천 환경에서 돌리는 것은 계산상 불가능합니다. 억지로 시도하면 프로젝트가 여기서 좌초합니다.

### 트랙 A — 학습용 축약 몰탈 모델 (전 환경, torch 벡터화)

몰탈 베드를 입자가 아닌 **두께 필드(thickness field)** 로 표현합니다.

```
상태: bed_thickness[num_envs, num_slots, K]   # 슬롯당 K=8 샘플 셀
      bed_coverage [num_envs, num_slots]

① 도포  : 노즐이 셀 반경 r 내 + 압출 on → thickness[cell] += rate·dt
② 착좌  : 벽돌 하면이 셀을 누른 깊이만큼 두께 감소,
          감소분 = 스퀴즈아웃 부피로 이월 (초과 시 페널티)
③ 반력  : 벽돌–하부단 접촉을 PhysX 컴플라이언트 재질로 근사
          (restitution 0, 고마찰, 낮은 contact stiffness)
④ 경화  : 착좌 후 안정 M step 경과 → 하부단과 fixed joint 생성
          (유한 break force = 미경화 몰탈의 green strength)
```

물리적으로 정확하진 않지만 **RL이 배워야 할 인과관계 — 안 바르면 안 붙는다 / 덜 누르면 두껍다 / 너무 누르면 삐져나온다 / 굳기 전엔 건드리면 밀린다 — 를 전부 보존**합니다. 학습에는 이걸로 충분합니다.

### 트랙 B — 데모용 고충실도 몰탈 (env 1~4개, 학습 미사용)

학습된 정책을 그대로 재생하면서 몰탈만 **PhysX PBD 입자(고점성 유체)** 로 교체해 압출·스퀴즈아웃·처짐을 실제 시뮬레이션하고 RTX 렌더로 촬영합니다. **데모 품질은 전적으로 여기서 나옵니다.**

### 도포 방식
- 좌완에 **압출 노즐 툴** 부착(양팔 분업). 트라울 문지르기는 접촉 난이도가 과도해 제외.
- **P5까지는 스크립트 궤적**(슬롯 시작→끝 직선 이송 + 압출 on/off), **P6에서 좌완 학습 액션으로 승격**.

---

## 5. MDP 설계

### 5.1 관측

**policy obs (약 140 dim)**

| 그룹 | 항목 | 차원 |
|---|---|---|
| 우완 | 관절 위치(기본자세 상대), 속도, 그리퍼 상태 | 6+6+1 |
| 좌완 | 관절 위치, 속도, 노즐 압출 상태 | 6+6+1 |
| 허리 | 3 관절 위치·속도 | 6 |
| 다리 (P5+) | 12 관절 위치·속도 | 24 |
| **균형 상태** | 골반 pose(높이·6D 회전), 선/각속도, **CoM 수평 위치(지지 다각형 정규화)**, ZMP 마진, 발 접촉 2, 발 F/T 12 | 7+6+2+1+2+12 |
| 중력 | 몸통 기준 중력 투영 벡터 | 3 |
| 우완 EE | pose(위치+6D), 선/각속도 | 9+6 |
| 대상 벽돌 | EE 기준 상대 위치·자세(6D), 파지 플래그 | 9+1 |
| 목표 슬롯 | EE 기준 상대 위치, **벽돌 기준 잔여 오차**(위치 3 + 6D) | 3+9 |
| 몰탈 | 목표 슬롯 두께 8샘플, 커버리지, 압출 on/off | 10 |
| 벽 상태 | 단·슬롯 인덱스 정규화, 직전 배치 벽돌 상대 위치, 진행률 | 6 |
| 접촉/힘 | finger 접촉 2, 우완 손목 F/T 6 | 8 |
| 이전 액션 | last action | 20 |

**critic obs (비대칭 actor-critic)** = policy obs + 특권 정보(벽돌 절대 pose, 마찰계수, 질량, 몰탈 실제 두께, 전 배치 벽돌 변위, 실제 CoM·ZMP, 외란력). 이족 균형 태스크에서 학습 안정화 효과가 특히 큽니다.

관측 노이즈: 위치 ±2 mm, 각도 ±1°, 힘 ±2 N, CoM ±5 mm 가우시안.

### 5.2 액션 (최대 20 dim, 단계별 활성화)

| 구간 | 차원 | 내용 | 활성 |
|---|---|---|---|
| 우완 | 6 | Task-space **상대 IK** delta (위치 0.04 m / 회전 0.04 rad) | P3~ |
| 우 그리퍼 | 1 | binary 개폐 | P3~ |
| **골반/CoM 명령** | 4 | 높이 delta, 피치 delta, 전후 shift, 좌우 shift → **1층 균형 정책 입력** | P5~ |
| 좌완 | 6 | 노즐 task-space delta | P6~ |
| 압출률 | 1 | [0, 1] | P6~ |
| 허리 | 2 | 피치·요 delta (스쿼트 시 상체 자세) | P5~ |

- 팔은 관절공간 직접 학습 대신 **task-space 상대 IK**. 조적은 mm 정밀도 태스크라 관절공간 학습은 샘플 효율이 치명적입니다.
- 다리 12 DoF는 **정책이 직접 제어하지 않습니다.** 골반 명령 4차원만 내면 동결된 1층이 다리를 처리합니다. 이 축소가 학습 성패를 가릅니다.
- 착좌 압착 구간만 P6에서 **OSC(힘 제어)** 교체 검토.

### 5.3 보상

```
r =  w1·reach + w2·grasp + w3·lift + w4·mortar_bed + w5·align
   + w6·seat_force + w7·joint_thickness + w8·release_stable
   - p1·FALL - p2·collapse - p3·com_margin - p4·foot_slip
   - p5·squeeze_out - p6·excess_force - p7·action_rate
   - p8·joint_vel - p9·self_collision - p10·posture_dev
```

| Term | 정의 | 가중치 |
|---|---|---|
| `reach` | `1 - tanh(d_ee_brick / 0.1)` | 1.0 |
| `grasp` | 양 finger 접촉 + 그리퍼 닫힘 | 2.0 |
| `lift` | 벽돌 높이 > 5 cm | 5.0 |
| `mortar_bed` | 베드 커버리지 × 두께 적정성(목표 12 mm 도포) | 8.0 |
| `align` | 파지 상태 `1 - tanh(d_brick_slot / 0.15)` + yaw 정렬 | 8.0 |
| `seat_force` | 착좌 중 수직력 [30, 80] N 밴드 유지 | 6.0 |
| `joint_thickness` | 최종 줄눈 8~12 mm | 15.0 |
| **`release_stable`** | 그리퍼 개방 후 30 step 벽돌 변위 < 3 mm | **30.0** |
| **`FALL`** | 전도 (몸통 높이 급락 or 기울기 > 40°) | **−200.0** |
| `collapse` | 기 배치 벽돌 변위 > 1 cm | −50.0 |
| `com_margin` | CoM이 지지 다각형 경계에 근접한 정도 | −5.0 |
| `foot_slip` | 접촉 중인 발의 수평 속도 | −2.0 |
| `squeeze_out` | 스퀴즈아웃 부피 초과분 | −5.0 |
| `excess_force` | 수직력 > 120 N 초과분 | −0.1/N |
| `posture_dev` | 기본 자세 대비 관절 편차 (기괴한 자세 억제) | −0.5 |
| `action_rate`, `joint_vel`, `self_collision` | 표준 정규화 페널티 | −0.01 ~ −5.0 |

> **설계 원칙 2가지.**
> ① `release_stable`이 조작 보상의 주축 — 놓는 순간이 아니라 **"놓고 나서도 안 무너지는가"** 로 성공을 정의해야 보상이 실제 조적 품질과 어긋나지 않습니다.
> ② `FALL` 페널티를 압도적으로 크게(−200) 둡니다. 전도는 에피소드 전체를 무효화하는 사건이라 다른 보상과 같은 스케일에 두면 정책이 "넘어져도 벽돌 하나 더 놓기"를 학습합니다.

### 5.4 종료 조건

- **성공**: `release_stable` + `joint_thickness` 동시 달성 → 슬롯 인덱스 전진(에피소드 유지)
- **실패**: **전도**(몸통 높이 < 기준의 60% 또는 기울기 > 40°) / 벽돌 낙하 / 벽 붕괴 / 수직력 > 200 N / 자기충돌 / 몰탈 미도포 착좌 / 발 미끄러짐 누적
- **타임아웃**: 벽돌 1장당 15초 (dt = 1/120, decimation 4 → 450 step). 스쿼트·자세 전환 시간이 들어가 Spot 대비 길게 잡음.

### 5.5 커리큘럼 (per-env 독립 난이도)

**핵심: `base_courses`를 12 → 0으로 내리며 균형 난이도를 올린다.** (§2 참조)

| 레벨 | 승급 조건 | `base_courses` | 변화 |
|---|---|---|---|
| L0 | 시작 | 12 (0.82 m) | **골반 고정**, 팔레트 스탠드 고정, 몰탈 자동 도포 |
| L1 | 성공률 70% | 12 | 팔레트 ±5 cm / ±15° 랜덤화 |
| L2 | 성공률 70% | 12 | 몰탈 스크립트 도포 실제 수행, 도포 품질이 결과에 반영 |
| L3 | 성공률 70% | 12 | 한 단 4장 순차 배치(이웃 벽돌 존재) |
| L4 | 성공률 70% | 12 | **골반 해제** — 동결된 1층 균형 정책 위에서 학습 |
| L5 | 성공률 70% | **12 → 6 → 3** | **작업면 하강** (0.82 → 0.41 → 0.21 m). 굴신·스쿼트 진입. 팔레트도 단계적으로 바닥으로 |
| **L5-STOP** | — | **3 (고정)** | **폴백 정지점** — L6 미달 시 여기서 동결하고 데모 조건으로 확정 (§11) |
| L6 | 성공률 70% | **3 → 0** | **바닥 시공** — 깊은 스쿼트. 3단 전체 자력 축조 + 어긋쌓기, 단 간 오차 누적 노출 |
| L7 | 성공률 70% | 0 | **양팔 몰탈 도포 학습 승격** + 도메인 랜덤화 풀가동 + 허용오차 강화(5 → 3 mm) |

> `base_courses`는 per-env 독립 값으로 두고, 환경마다 성공률에 따라 개별적으로 내립니다. 전 환경을 동시에 내리면 난이도 급변으로 정책이 무너집니다.

---

## 6. 씬(Scene) 구성

```
/World/envs/env_.*/
├── Robot            (Atlas + 양 손목 그리퍼/노즐, ArticulationCfg, 초기 = 기본 스탠스)
├── Ground           (콘크리트 슬래브, 마찰 0.9)
├── Pallet/Brick_src_[0..15]   (RigidObjectCollectionCfg, 스폰 랜덤화)
├── Wall/Brick_placed_[0..11]  (RigidObjectCollectionCfg, 미사용 시 지하 대기 후 텔레포트)
├── MortarNozzle     (좌완 EE 고정 조인트, 트랙 A는 시각 전용)
├── ContactSensor    (finger ×2, 발바닥 ×2, 벽돌 하면)
└── FrameTransformer (ee_frame ↔ brick ↔ slot 마커, CoM 추적)
```

**물리 튜닝 — 초기 실패 대부분이 여기서 발생합니다.**

| 파라미터 | 값 | 이유 |
|---|---|---|
| `dt` | 1/120 | 스택 태스크에서 큰 dt는 즉시 붕괴 |
| `solver_position_iteration_count` | 16~24 | 다층 스택 + 이족 접촉 안정성 |
| `solver_velocity_iteration_count` | 1 | 위치 반복 우선 |
| 벽돌 마찰 | static 0.9 / dynamic 0.8, restitution 0.0 | 몰탈 접촉 근사 |
| **발바닥 마찰** | **static 1.0 / dynamic 0.9** | 미끄러짐이 곧 전도. 랜덤화 하한도 0.6 이상 |
| `contact_offset` / `rest_offset` | 0.002 / 0.0005 | 과도한 offset은 "떠 있는 벽" 유발 |
| `max_depenetration_velocity` | 1.0 | 관통 복구 폭발 방지 |
| 충돌 형상 | **box / convex hull** (삼각메시 아님) | 성능·안정성 모두 유리. Atlas 원본 메시는 반드시 단순화 |
| CCD | off | 저속 조작 태스크에 불필요 |

---

## 7. 리포지토리 구조

Isaac Lab 본체를 포크하지 않고 **외부 확장(external extension)** 으로 갑니다 (`./isaaclab.sh --new`).

```
masonry_rl/
├── source/masonry_rl/masonry_rl/
│   ├── robots/
│   │   ├── preset.py               # RobotPresetCfg — 로봇 독립 인터페이스
│   │   ├── atlas.py                # Atlas v5 + 그리퍼 (1순위)
│   │   └── g1.py                   # Unitree G1 폴백 프리셋
│   ├── tasks/
│   │   ├── balance/                # 1층: 균형 정책 (사전 학습·동결)
│   │   │   ├── balance_env_cfg.py
│   │   │   └── agents/
│   │   └── masonry/                # 2층: 조작 정책 (본 산출물)
│   │       ├── __init__.py         # Isaac-Masonry-Wall-Atlas-IK-Rel-v0
│   │       ├── masonry_env_cfg.py
│   │       ├── mdp/
│   │       │   ├── observations.py # 슬롯 잔여오차, 몰탈 두께, CoM/ZMP
│   │       │   ├── rewards.py      # align/seat_force/joint_thickness/release_stable
│   │       │   ├── terminations.py # 전도·붕괴·과대력
│   │       │   ├── actions.py      # 골반 명령 → 동결 균형 정책 브리지
│   │       │   ├── events.py
│   │       │   └── curriculums.py
│   │       ├── mortar/
│   │       │   ├── reduced_model.py  # 트랙 A: 두께 필드 (torch 벡터화)
│   │       │   └── pbd_demo.py       # 트랙 B: 데모용 PBD 입자
│   │       ├── wall_planner.py
│   │       └── agents/rsl_rl_ppo_cfg.py
│   └── assets/                     # brick.usd, pallet.usd, nozzle.usd, gripper.usd
├── scripts/
│   ├── train.py / play.py
│   ├── train_balance.py            # 1층 사전 학습
│   ├── test_stack_stability.py     # P1 게이트
│   ├── eval_wall.py                # 벽 완성도 배치 평가
│   └── record_demo.py              # 트랙 B 렌더 + 영상 캡처
├── configs/
└── docs/
```

---

## 8. 학습 설정 — RTX A6000 48 GB

### 8.1 병렬 환경 수

Atlas는 30 DoF라 Spot(18 DoF) 대비 환경당 비용이 큽니다. **VRAM보다 접촉·관절 연산이 병목**입니다.

| 단계 | `num_envs` | 근거 |
|---|---|---|
| P1~P2 (디버깅, GUI) | **16~32** | 시각 확인용 |
| P4 (1층 균형 정책, 벽돌 없음) | **4096** | 강체가 로봇뿐이라 가장 가벼움 |
| P3 (골반 고정, 상체만 ~17 DoF 활성) | **2048** | 다리 비활성으로 비용 절감 |
| P5~P6 (전신 + 벽돌 16 + 몰탈) | **512** | 30 DoF + 접촉쌍 급증. 여기가 가장 무거움 |
| P7 (데모 렌더) | **1~4** | RTX 렌더 + PBD 입자 |

### 8.2 PhysX GPU 버퍼 — **필수 조정**

벽돌 스택 + 이족 접촉 조합은 기본 버퍼를 넘겨 `PxgDynamicsMemoryConfig` 오버플로로 죽습니다. 학습 시작 전 반드시 상향하십시오.

```python
sim = SimulationCfg(
    dt=1/120, render_interval=4,
    physx=PhysxCfg(
        solver_type=1,
        max_position_iteration_count=24,
        max_velocity_iteration_count=1,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**20,
        gpu_found_lost_pairs_capacity=2**22,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**22,
        gpu_collision_stack_size=2**28,
        gpu_heap_capacity=2**26,
        gpu_temp_buffer_capacity=2**24,
        gpu_max_num_partitions=8,
    ),
)
```
(정확한 필드명은 설치된 Isaac Lab 버전에서 확인 후 확정 — P0 항목)

### 8.3 PPO 하이퍼파라미터 (rsl_rl)

```
algorithm         : PPO
num_envs          : 2048 (P3) / 4096 (P4 균형) / 512 (P5~P6)
num_steps_per_env : 32  (P5~P6는 48 — env 수가 적어 배치 크기 보전)
mini_batches      : 4
learning_epochs   : 5
learning_rate     : 1.0e-3, adaptive (desired_kl = 0.01)
gamma             : 0.99      lambda : 0.95
entropy_coef      : 0.005     clip   : 0.2
max_grad_norm     : 1.0       value_loss_coef : 1.0
network           : MLP [512, 256, 128], ELU, 비대칭 actor-critic
empirical_normalization : True
max_iterations    : 4000 ~ 8000 (단계별)
```

관측 약 140차원 + 이족 균형 태스크라 네트워크를 **[512, 256, 128]** 로 둡니다. 512 env 단계에서는 `num_steps_per_env`를 48로 올려 업데이트당 샘플 수(512×48 ≈ 24.6k)를 확보합니다.

### 8.4 예상 학습 시간 (A6000 48 GB 단일)

| 단계 | 예상 처리량 | 총 스텝 | 예상 wall-clock |
|---|---|---|---|
| P4 (1층 균형, 4096 env, 벽돌 없음) | 40~70k FPS | 1.5e8 | **1~1.5 시간** |
| P3 (상체 단독, 2048 env) | 12~20k FPS | 2.0e8 | **3~5 시간** |
| P5 (전신 통합 + 스쿼트, 512 env) | 4~7k FPS | 3.0e8 | **12~21 시간** |
| P6 (3단 + 양팔 + DR, 512 env) | 3~5k FPS | 4.0e8 | **22~37 시간** |

**P6는 1~2일 연속 실행입니다.** 체크포인트를 200 iteration마다 저장해 중단·재개가 가능하게 하고, 일정이 급하면 3단→2단 축소 또는 스텝 수 절감으로 트레이드오프합니다. 학습은 반드시 `--headless`, 영상은 `play.py --enable_cameras` 별도 실행.

### 8.5 운영 팁

- 로깅: TensorBoard 기본 + W&B 선택. **커리큘럼 레벨 분포와 전도율을 반드시 로깅** — 레벨이 안 오르거나 전도율이 안 떨어지면 조기에 발견해야 합니다.
- 체크포인트: 200 iteration마다 저장, 성공률 기준 best 별도 보관.
- A6000은 fp32로 충분(PhysX 자체가 fp32). AMP 이득 미미하므로 생략.
- 학습 중 다른 GUI 세션 금지 — VRAM 경합으로 접촉 버퍼가 터집니다.
- Atlas 원본 메시가 무거우면 시각 메시와 충돌 메시를 분리(충돌은 convex hull)해야 FPS가 나옵니다.

---

## 9. 로드맵

| Phase | 목표 | 완료 게이트 | 예상 |
|---|---|---|---|
| **P0. 자산 확보** | Isaac Lab 설치·버전 고정, **`robot_descriptions`로 DRC Atlas URDF 확보 → 임포트·검수·그리퍼 부착**, 로봇 독립 프리셋 인터페이스 설계 | Atlas가 서서 팔로 물체 파지 (스크립트 데모) | **5~8일** |
| **P1. 씬 & 물리** | 벽돌/팔레트/노즐 자산, 씬 cfg, 물리·GPU 버퍼 튜닝, 메시 단순화 | 3단 벽 수동 배치 후 10초 무붕괴, 512 env 버퍼 오버플로 없음 | 4~5일 |
| **P2. MDP 구현** | obs/action/reward/termination + 축약 몰탈 모델 | 랜덤 정책 구동, 보상 sanity check, 두께 필드 검증 | 5~7일 |
| **P3. 상체 단독 학습** | 골반 고정, 허리 높이, L0~L3 | 허리 높이 1단 4장 성공률 ≥ 85% | 5~7일 |
| **P4. 균형 정책** | 1층 사전 학습(제자리 외란 저항 + 골반 명령 추종) | 팔 반력 외란 하 전도 0%, 골반 명령 추종 오차 ≤ 2 cm | 4~6일 |
| **P5. 통합 + 스쿼트** | 동결 1층 위 조작 학습, L4~L5(높이 하강) | 바닥 1단 성공률 ≥ 80%, 전도율 ≤ 1% | **2~3주** |
| **P6. 3단 + 양팔 + 강건화** | L6~L7, 몰탈 도포 학습 승격, 도메인 랜덤화 | 벽 완성률 ≥ 70%, 줄눈 유지율 ≥ 80% | **3~4주** |
| **P7. 데모 산출** | 트랙 B PBD 몰탈, 카메라 리그, 렌더 영상 (선택: 5단 확장) | 고품질 영상 + 평가 리포트 | 5~7일 |

**총 예상: 8~13주.** Spot 안(6~9주) 대비 2~4주 늘어나며, 증가분의 대부분은 **P5(이족 균형 + 스쿼트 통합)** 입니다. P0는 §1 조사로 자산 경로가 확정되어 당초 1~2주에서 5~8일로 단축했습니다.

---

## 10. 평가 지표

| 지표 | 정의 |
|---|---|
| Placement success rate | 허용오차 내 착좌 + 안정 유지 비율 |
| Position / Yaw RMSE | 목표 슬롯 대비 최종 오차 |
| Joint thickness distribution | 줄눈 두께 분포 (8~12 mm 유지율) |
| Course levelness | 단별 상면 수평도 누적 오차 |
| Collapse rate | 기 배치 벽돌 붕괴 비율 |
| **Fall rate** | 에피소드당 전도 발생률 (Atlas 최우선 안전 지표) |
| **CoM margin (min)** | 에피소드 중 CoM–지지경계 최소 거리 |
| **Squat depth range** | 달성한 골반 높이 범위 (전신 가동 범위 입증) |
| Peak seating force | 착좌 시 최대 수직력 |
| Cycle time | 벽돌 1장당 소요 시뮬 시간 |
| Wall completion | 12장 중 성공 배치 비율 |

`eval_wall.py`로 100 에피소드 배치 평가 + 실패 케이스 영상 자동 저장.

---

## 11. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **전기 Atlas 자산 부재** | 중간 (조사 완료로 하향) | §1 확정 — 전기 Atlas 공개 자산은 없음. **DRC Atlas를 `robot_descriptions`로 즉시 확보**(A안). 로봇 독립 프리셋으로 G1 폴백 상시 가능 |
| **DRC Atlas URDF 품질 (구형·손 없음·175 kg)** | 중간 | P0에서 관절 리밋·관성 텐서 검수, 그리퍼 부착, 충돌 메시 convex hull 단순화. 게인은 균형 정책 학습(P4)에서 실측 기반 재튜닝 |
| **이족 균형 실패 — 스쿼트 중 전도** | **높음 · Atlas 고유** | ① 1층 균형 정책 사전 학습·동결(계층화) ② `base_courses` 12→0 하강 커리큘럼 ③ `FALL` −200 페널티 ④ 최후엔 "무릎 꿇은 자세"로 골반 고정 |
| **바닥 시공(`base_courses`=0)이 끝내 안 되는 경우** | 높음 | **확정 폴백: `base_courses`=3에서 정지.** 이미 3단이 쌓인 벽 위에서 4~6단을 시공하는 조건으로 데모를 확정한다(§11.1) |
| **mm 정밀도 미달 (부유 상체)** | 높음 | 착좌 마지막 2 cm를 impedance 기반 스크립트 삽입 동작으로 위임하는 하이브리드 |
| **몰탈 모델 과욕** | 높음 | 학습에는 축약 모델만. PBD는 데모 전용으로 격리(§4). **이 경계를 넘지 말 것** |
| **희소 보상 탐색 실패** | 중간 | Isaac Lab Mimic으로 소수 텔레오퍼레이션 데모 → 자동 증강 → BC 사전학습 후 PPO 파인튜닝 |
| **오차 누적으로 3단 붕괴** | 중간 | `joint_thickness`·`course levelness` 보상 강화, 하부 단 확정 후 fixed joint 고정 |
| **학습 시간 과다 (P6 1~2일)** | 중간 | 체크포인트 재개 지원, 3단→2단 축소 옵션, 512→1024 env 상향 실험 |
| **일정 지연** | 중간 | P5까지가 핵심 가치(이족 휴머노이드가 바닥에서 조적). P6·P7은 축소 가능 |

### 11.1 확정 폴백 — 3단 위 시공 (`base_courses = 3`)

바닥 시공(L6)이 끝내 수렴하지 않을 경우의 **대체 데모 조건을 미리 확정해 둡니다.** 폴백을 나중에 급하게 정하면 평가 기준과 씬 자산을 다시 만들어야 하므로, 처음부터 같은 파라미터로 구현합니다.

```
조건    : base_courses = 3  (이미 3단이 쌓인 벽 위에서 4~6단 시공)
작업면  : 바닥 +211 mm → 시공 완료 시 상단 +402 mm
사전분  : 하부 3단은 kinematic 강체로 스폰 (경화 완료 가정)
          → 붕괴 위험 없음, 접촉 연산 부담 없음, 오차 누적 없음
평가    : 벽 완성률은 "신규 12장" 기준으로 동일하게 산정 (지표 정의 불변)
```

**구현상 이점:** `base_courses`가 이미 커리큘럼 파라미터(§5.5)라, 폴백 전환에 **코드 변경이 필요 없습니다.** L5-STOP에서 커리큘럼을 동결하고 그 정책으로 평가·렌더링하면 그대로 데모가 됩니다.

> **폴백이 실제로 사주는 것 — 자세 해석으로 정정 (v0.5).**
> 초안에서는 "3단이 스쿼트 깊이를 절반으로 줄여준다"고 적었으나, `scripts/export_eatlas_viewer.py`의 자세 해석 결과 **틀린 서술이었습니다.** 실측:
>
> | `base_courses` | 작업면 | 골반 높이 | 스쿼트 깊이 | 몸통 굽힘 |
> |---|---|---|---|---|
> | 12 | 0.814 m | 0.967 m | 0.040 m | 0.0° |
> | 6 | 0.412 m | 0.561 m | 0.446 m | 0.0° |
> | **3** | 0.211 m | **0.412 m (무릎 한계)** | 0.595 m | **19.5°** |
> | **0** | 0.010 m | **0.412 m (무릎 한계)** | 0.595 m | **43.5°** |
>
> **3단이든 0단이든 스쿼트는 똑같이 무릎 한계까지 내려갑니다.** 3단이 사주는 것은 **몸통 굽힘 43.5° → 19.5°** 입니다.
>
> 그런데 이 이득이 더 큽니다. 이족 전도 위험을 직접 키우는 것은 스쿼트 깊이가 아니라 **몸통 굽힘** — CoM을 지지 다각형 앞쪽 경계로 밀어내기 때문입니다(§5.3 `com_margin`). 폴백의 가치는 유지되나 근거가 다릅니다.
>
> **더 근본적인 제약도 함께 확인됐습니다: 몸통을 세운 채로는 바닥 단에 손이 닿지 않습니다.** 팔 리치로는 어깨가 너무 멀어, 무릎을 한계까지 접고 몸통까지 굽혀야 겨우 도달합니다. `base_courses=0`은 무릎 한계와 몸통 굽힘을 **동시에 최대로 사용하는** 구성이며, 이것이 §11의 "이족 균형 실패" 리스크가 높음인 정량적 근거입니다.
>
> 그래서 `base_courses`를 **연속 파라미터로 구현**해 두었습니다 — 6단(0.41 m)은 몸통 굽힘 0°에 스쿼트만 요구하므로, 균형이 근본적으로 안 풀릴 때의 다음 정지점으로 3단보다 오히려 명확합니다. 어느 높이에서 멈출지는 P5 결과를 보고 판단하면 됩니다.

---

## 12. Boston Dynamics 공개 사례에서 가져올 설계 시사점

§1.1에서 확인한 BD의 공개 사례는 우리 계획의 검증이자 참고자료입니다.

| BD 사례 | 우리 계획에 주는 시사점 |
|---|---|
| **약 45 kg 화물을 전신 RL로 들어올림** | 무거운 물체 + 이족 균형 조합이 RL로 실제로 풀린다는 실증. 우리 벽돌은 2.4 kg으로 훨씬 가벼워, **난이도의 본질은 무게가 아니라 mm 단위 배치 정밀도**임이 분명해짐 → 보상 설계를 정밀도 쪽에 집중시키는 근거 |
| **whole-body(전신 단일) 정책 채택** | 우리는 계층형(균형 동결 + 조작)을 택했음. BD 방식이 성능 상한은 높지만 **단일 GPU·수개월 일정에는 계층형이 현실적**. P6까지 성공 후 여력이 있으면 전신 단일 정책을 P7 확장 실험으로 시도 |
| **zero-shot sim-to-real 전이 성공** | 본 프로젝트는 시뮬 데모까지가 스코프라 직접 필요는 없으나, **도메인 랜덤화(L7)를 제대로 넣어두면 후속 실기 확장 경로가 열림**. L7을 축소 대상에서 제외하는 근거 |
| **Isaac Lab을 실제 프로덕션에 사용** | 툴 선택이 검증됨. Isaac Lab + rsl_rl 조합 유지 |
| **양산형 56 DoF** | DRC Atlas(약 30 DoF)와 큰 차이. 향후 전기 Atlas 자산 공개 시 관측·액션 차원이 커지므로, **`RobotPresetCfg`가 DoF 수에 의존하지 않도록 설계** (관절 이름 그룹 기반 인덱싱) |

---

## 부록 A. 참고 Isaac Lab 레퍼런스

| 레퍼런스 | 재사용 포인트 |
|---|---|
| Unitree H1 / G1 휴머노이드 태스크 | 휴머노이드 액추에이터 게인, 균형·보행 보상 구조, 관측 설계 |
| `Isaac-Velocity-Flat-*` 계열 | 지형·접촉 센서·도메인 랜덤화 이벤트 패턴 |
| `Isaac-Lift-Cube-Franka-v0` | 보상 shaping 구조, IK 액션 골격 |
| `Isaac-Stack-Cube-Franka-IK-Rel-v0` | 다중 물체 스택, 단계형 종료, 데모 수집 파이프라인 |
| Isaac Lab Mimic | 소수 데모 → 대량 데이터 생성 (탐색 실패 대응책) |

> API 시그니처는 설치 버전에 따라 다릅니다. P0에서 버전을 태그로 고정하고, 위 태스크 소스를 실제로 열어 현재 버전 API에 맞춰 본 문서의 클래스·필드명을 확정합니다.

## 부록 B. P0 착수 체크리스트

- [ ] Isaac Sim / Isaac Lab 설치, 버전 태그 고정, `docs/setup.md` 작성
- [ ] A6000 드라이버·CUDA 확인, 기본 휴머노이드 태스크 학습 재현 (FPS 실측 → §8.4 추정치 보정)
- [ ] `pip install robot_descriptions` → `atlas_drc_description` / `atlas_v4_description` **양쪽 다 받아 비교**, 관절 수·메시 품질 좋은 쪽 채택
- [ ] 각 원본 리포지토리(Drake / roboschool) **라이선스 조건 확인** → `docs/asset-notes.md` 기록
- [ ] URDF Importer 변환, **관절 리밋·질량 관성 텐서 검수**, 충돌 메시 convex hull 단순화
- [ ] 우완 손목에 평행 그리퍼 / 좌완에 몰탈 노즐 부착
- [ ] `RobotPresetCfg` 인터페이스 정의 (**DoF 수 비의존, 관절 이름 그룹 기반**) + G1 폴백 프리셋 동시 작성
- [ ] `./isaaclab.sh --new` 확장 스캐폴딩 생성
- [ ] **Atlas가 선 자세로 물체 파지 — 스크립트 데모 성공 (P0 게이트)**
