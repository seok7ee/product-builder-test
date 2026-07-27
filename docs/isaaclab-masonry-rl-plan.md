# Isaac Lab 조적(Masonry / Bricklaying) 강화학습 프로젝트 계획서

> 상태: **v0.2 (조건 확정본)** · 대상: Isaac Lab 2.x (Isaac Sim 4.5+) · 목표: 시뮬레이션 데모까지

## 확정된 프로젝트 조건

| 항목 | 확정 내용 |
|---|---|
| 로봇 | **Boston Dynamics Spot + Spot Arm** (6-DoF 팔 + 그리퍼, 12-DoF 다리) |
| 이동 | **제자리 고정** — 보행 없음, 몸체 자세(높이/피치/요) 제어만 사용 |
| 작업 | **바닥에서부터 소형 벽 축조** (4장 × 3단, 약 790 × 211 mm) |
| 몰탈 | **포함** — 도포(압출) → 압착 착좌 → 경화 전 과정 모델링 |
| 최종 산출물 | **시뮬레이션 데모** (학습 정책 + 렌더링 영상), 실기 이식 없음 |
| 학습 하드웨어 | **NVIDIA RTX A6000 (Ampere, 48 GB)** 단일 GPU |

> GPU 표기 확인: "RTX PRO A6000"을 **RTX A6000 48 GB**로 가정하고 세팅했습니다. 만약 신형 **RTX PRO 6000 Blackwell (96 GB)** 이라면 §8의 `num_envs`를 그대로 2배로 올리면 됩니다(그 외 설정 동일).

---

## 0. 한 줄 정의

제자리에 선 **Spot이 몸체 자세를 조절해 가며, 바닥의 팔레트에서 벽돌을 집어 → 몰탈 베드를 압출 도포 → 벽돌을 목표 슬롯에 정렬·압착 착좌 → 놓기**를 반복해 3단 소형 벽을 완성하는 정책을, Isaac Lab GPU 병렬 시뮬레이션에서 PPO로 학습하고 데모 영상으로 산출한다.

**최종 성공 기준**
- 벽 완성률(12장 중 배치 성공) ≥ 70%
- 장당 배치 성공률 ≥ 85% — 위치 오차 ≤ 5 mm, yaw 오차 ≤ 2°
- 줄눈 두께 8~12 mm 유지율 ≥ 80%
- 기 배치 벽돌 붕괴율 ≤ 5%
- 데모: 1~2대 환경 고품질 렌더 영상(몰탈 압출/스퀴즈아웃 시각화 포함)

---

## 1. Spot 특유의 설계 제약 — 이 프로젝트의 진짜 난이도

고정 베이스 산업용 팔(Franka/UR)과 Spot은 조적 태스크에서 **근본적으로 다른 문제**입니다. 이 3가지가 계획 전체를 좌우합니다.

**(1) 베이스가 떠 있다.** Spot의 몸체는 4개 다리로 지지되는 컴플라이언트 부유 베이스입니다. 팔이 벽돌(약 2.5 kg)을 뻗어 들거나 몰탈을 압착(30~80 N)하면 **반력으로 몸이 흔들리고 기웁니다.** 고정 베이스라면 무시할 오차가 여기서는 배치 정밀도를 직접 깎아먹습니다. → 그래서 학습을 **베이스 고정(P3) → 자세 제어기 위 학습(P4) → 전신 협조(P5)** 3단으로 나눕니다.

**(2) 작업 반경이 좁다.** Spot Arm의 최대 리치는 약 0.98 m, 가반하중 약 11 kg입니다. 벽돌 무게는 문제없지만 **제자리 고정이라 벽 크기와 팔레트 위치가 리치 안에 강제로 들어와야 합니다.** 그래서 벽 규격을 790 mm로 확정하고, 몸체 자세(높이 ±0.15 m, 피치 ±20°, 요 ±30°)를 **액션에 포함**해 유효 작업공간을 넓힙니다. 바닥 1단과 3단 상부를 같은 자세로 다 닿기 어렵기 때문에, 자세 제어는 선택이 아니라 필수입니다.

**(3) 몰탈이 태스크를 2단계로 늘린다.** 무몰탈 건식 적재와 달리, 각 벽돌마다 **베드 도포 → 착좌 압착**이 붙습니다. 도포 품질이 나쁘면 다음 벽돌의 수평이 무너지므로 **오차가 누적**됩니다. 몰탈은 보상 항목이자 상태 변수입니다.

---

## 2. 자산(Asset) 확보 전략 — P0의 최우선 과제

`Isaac Lab`에는 Spot **보행** 자산·태스크(`SPOT_CFG`, `Isaac-Velocity-Flat-Spot-v0`)가 기본 포함되어 있지만, **팔이 달린 Spot(Spot Arm) 자산은 버전에 따라 없을 수 있습니다.** P0에서 아래 순서로 확인·확보합니다.

1. **1순위** — Isaac Sim 자산 서버의 Spot with Arm USD 존재 여부 확인 (`Isaac/Robots/BostonDynamics/` 경로). 있으면 관절 리밋·액추에이터 게인만 재설정해 사용.
2. **2순위** — Boston Dynamics 공개 `spot_description` URDF(팔 포함)를 Isaac Lab **URDF Importer**로 변환 후 `ArticulationCfg` 직접 작성. 액추에이터는 `ImplicitActuatorCfg`로 시작(팔: stiffness 400 / damping 40, 다리: 기존 `SPOT_CFG` 값 계승).
3. **3순위(폴백)** — 기본 Spot USD에 6-DoF 팔을 **직접 부착**(몸체 상단 고정 조인트 + 팔 링크 체인). 치수는 공개 스펙(리치 0.98 m) 기준으로 근사.

> **P0 게이트:** 3가지 경로 중 하나로 "Spot이 서서 팔을 뻗어 바닥 물체를 집는다"가 확인되기 전에는 P1로 넘어가지 않습니다. 여기서 막히면 전체 일정이 밀리므로 리스크 1순위입니다.

> Atlas는 시뮬레이션 가능한 공개 URDF/USD가 없어 이 프로젝트 대상에서 제외했습니다. Spot Arm 전제로 진행합니다.

---

## 3. 태스크 분해 — 학습 단위는 "벽 1개"가 아니라 "벽돌 1장"

조적은 long-horizon + contact-rich + 희소 보상이 겹친 조합이라 벽 전체를 end-to-end로 학습하면 거의 확실히 실패합니다. 따라서:

- **학습 단위** = "지정된 슬롯 1곳에 몰탈 도포 후 벽돌 1장을 안전하게 착좌시키기"
- **슬롯 순서**(1단 좌→우, 2단 반턱 어긋쌓기, 3단…)는 학습이 아닌 **결정론적 플래너**(`wall_planner.py`)가 결정
- 기 배치 벽돌은 관측·충돌 대상 + 붕괴 페널티 소스로만 존재

정책은 "벽을 쌓는 법"이 아니라 "벽돌 1장을 잘 놓는 법"을 배웁니다. **이 분리가 설계의 핵심축입니다.**

### 벽 사양 (Spot 리치 내 확정)

```
벽돌: 표준형 190 × 90 × 57 mm, 질량 2.4 kg (±10% 랜덤화)
줄눈: 목표 10 mm (허용 8~12 mm)
1단  : 4장 → 길이 4×190 + 3×10 = 790 mm
단 높이: 57 + 10 = 67 mm → 3단 = 201 mm (+ 바닥 베드 10 mm = 211 mm)
총 벽돌: 12장 (+ 어긋쌓기용 반절 벽돌 4장)
배치   : 벽 중심선 Spot 전방 0.65 m, 팔레트 우측 0.45 m (둘 다 리치 0.98 m 내)
```

---

## 4. 몰탈 모델링 — 2트랙 전략

**핵심 판단: 학습용과 데모용 몰탈 모델을 분리합니다.** PBD 입자 유체를 1024개 환경에서 돌리는 것은 계산상 불가능합니다. 억지로 시도하면 프로젝트가 여기서 좌초합니다.

### 트랙 A — 학습용 "축약 몰탈 모델" (Reduced-order, 전 환경 적용)

몰탈 베드를 **입자가 아니라 두께 필드(thickness field)** 로 표현하고, 전부 torch 텐서 연산으로 GPU 벡터화합니다.

```
상태: bed_thickness[num_envs, num_slots, K]     # 슬롯당 K=8 샘플 셀
      bed_coverage[num_envs, num_slots]          # 도포된 셀 비율

① 도포(extrude): 노즐이 셀 위 반경 r 안 + 압출 on → thickness[cell] += rate·dt
② 착좌(seat)   : 벽돌 하면이 셀을 누른 깊이만큼 thickness 감소,
                 감소분 = 스퀴즈아웃 부피로 이월(초과 시 페널티)
③ 접촉 반력    : 벽돌–하부단 접촉을 PhysX 컴플라이언트 재질로 근사
                 (restitution 0, 높은 마찰, 낮은 contact stiffness)
④ 경화(cure)   : 착좌 후 안정 유지 M step 경과 → 벽돌–하부단 fixed joint 생성
                 (break force 유한값 = 미경화 몰탈의 'green strength')
```

이 모델은 물리적으로 정확하진 않지만, **RL이 배워야 할 인과관계(도포 안 하면 못 붙는다 / 덜 누르면 두껍다 / 너무 누르면 삐져나온다 / 굳기 전엔 건드리면 밀린다)를 전부 보존**합니다. RL 학습에는 이 정도면 충분합니다.

### 트랙 B — 데모용 고충실도 몰탈 (환경 1~4개, 학습 미사용)

학습된 정책을 그대로 재생하면서, 몰탈만 **PhysX PBD 입자(고점성 유체)** 로 교체해 압출·스퀴즈아웃·처짐을 실제로 시뮬레이션하고 RTX 렌더로 촬영합니다. 정책은 축약 모델로 학습됐지만, 착좌 궤적이 물리적으로 타당하면 입자 몰탈에서도 그럴듯하게 동작합니다. 데모 품질은 여기서 나옵니다.

### 몰탈 도포 동작

- 엔드이펙터에 **압출 노즐 툴** 부착 가정(그리퍼 병용). 트라울 문지르기(troweling)는 접촉 난이도가 과도해 제외.
- **P4에서는 스크립트 궤적**(슬롯 시작→끝 직선 이송 + 압출 on/off)으로 시작, **P5에서 학습 액션으로 승격**(노즐 높이·속도·압출률을 정책이 결정).

---

## 5. MDP 설계

### 5.1 관측

**policy obs (약 95 dim)**

| 그룹 | 항목 | 차원 |
|---|---|---|
| 팔 고유수용 | 관절 위치(기본자세 상대), 관절 속도, 그리퍼 상태 | 6+6+1 |
| 몸체 상태 | 몸체 높이·롤·피치·요, 선속도·각속도, 중력 투영 벡터 | 4+6+3 |
| 다리 (P4+) | 12 관절 위치, 발 접촉 4 | 12+4 |
| EE | EE pose(위치 + 6D 회전), 선/각속도 | 9+6 |
| 대상 벽돌 | EE 기준 상대 위치·자세(6D), 파지 플래그 | 9+1 |
| 목표 슬롯 | EE 기준 상대 위치, **벽돌 기준 잔여 오차**(위치 3 + 6D 자세) | 3+9 |
| 몰탈 | 목표 슬롯 베드 두께 8샘플, 커버리지, 압출 on/off | 8+1+1 |
| 벽 상태 | 단 번호·슬롯 인덱스 정규화, 직전 배치 벽돌 상대 위치, 진행률 | 6 |
| 접촉/힘 | finger 접촉력 2, 손목 F/T 6(clip) | 8 |
| 이전 액션 | last action | 11 |

**critic obs (비대칭 actor-critic)** = policy obs + 특권 정보(벽돌 절대 pose, 마찰계수, 질량, 몰탈 실제 두께, 모든 배치 벽돌 변위, 몸체 외란력). 부유 베이스 태스크에서 학습 안정화 효과가 큽니다.

관측 노이즈: 위치 ±2 mm, 각도 ±1°, 힘 ±2 N 가우시안.

### 5.2 액션 (11 dim)

| 구간 | 차원 | 내용 |
|---|---|---|
| 팔 | 6 | Task-space **상대 IK** delta — 위치 0.04 m / 회전 0.04 rad 스케일 |
| 그리퍼 | 1 | binary 개폐 |
| 몸체 자세 | 3 | 높이 delta, 피치 delta, 요 delta → 자세 제어기 목표값 (P4+) |
| 몰탈 압출 | 1 | 압출률 [0,1] (P5에서 활성, P4까지는 스크립트) |

- 팔은 관절공간 직접 학습 대신 **task-space 상대 IK** 사용. 조적은 mm 단위 위치 정밀도 태스크라 관절공간 학습은 샘플 효율이 치명적으로 나쁩니다.
- 압착 착좌 구간만 P5에서 **OSC(Operational Space Control) 힘 제어**로 교체 검토.

### 5.3 보상

```
r =  w1·reach + w2·grasp + w3·lift + w4·mortar_bed + w5·align
   + w6·seat_force + w7·joint_thickness + w8·release_stable
   - p1·collapse - p2·squeeze_out - p3·body_instability
   - p4·excess_force - p5·action_rate - p6·joint_vel - p7·self_collision
```

| Term | 정의 | 가중치 |
|---|---|---|
| `reach` | `1 - tanh(d_ee_brick / 0.1)` | 1.0 |
| `grasp` | 양 finger 접촉 + 그리퍼 닫힘 | 2.0 |
| `lift` | 벽돌 높이 > 5 cm | 5.0 |
| `mortar_bed` | 목표 슬롯 베드 커버리지 × 두께 적정성(목표 12 mm 도포) | 8.0 |
| `align` | 파지 상태 `1 - tanh(d_brick_slot / 0.15)` + yaw 정렬 | 8.0 |
| `seat_force` | 착좌 중 수직력이 [30, 80] N 밴드 내 유지 | 6.0 |
| `joint_thickness` | 최종 줄눈 두께가 8~12 mm 내 | 15.0 |
| **`release_stable`** | 그리퍼 개방 후 30 step 동안 벽돌 변위 < 3 mm | **30.0** |
| `collapse` | 기 배치 벽돌 변위 > 1 cm | −50.0 |
| `squeeze_out` | 스퀴즈아웃 부피 초과분 | −5.0 |
| `body_instability` | 몸체 롤·피치 |각도| 및 각속도 | −2.0 |
| `excess_force` | 수직력 > 120 N 초과분 | −0.1/N |
| `action_rate`, `joint_vel`, `self_collision` | 표준 정규화 페널티 | −0.01 ~ −5.0 |

> **설계 원칙: `release_stable`이 최종 보상의 주축.** 놓는 순간이 아니라 **"놓고 나서도 안 무너지는가"** 로 성공을 정의해야 보상이 실제 조적 품질과 어긋나지 않습니다. `joint_thickness`가 그 다음인 이유는 줄눈 오차가 상단 단으로 누적되기 때문입니다.

### 5.4 종료 조건

- **성공**: `release_stable` + `joint_thickness` 동시 달성 → 슬롯 인덱스 전진(에피소드 유지)
- **실패**: 벽돌 낙하 / 붕괴 감지 / 수직력 > 200 N / 몸체 롤·피치 > 35°(전도) / 팔 자기충돌 / 몰탈 미도포 상태 착좌
- **타임아웃**: 벽돌 1장당 12초 (dt = 1/120, decimation 4 → 360 step). 몰탈 도포가 있어 무몰탈 대비 길게 잡음.

### 5.5 커리큘럼 (per-env 독립 난이도)

| 레벨 | 승급 조건 | 변화 |
|---|---|---|
| L0 | 시작 | 베이스 고정, 팔레트 위치 고정, 슬롯 1개(1단), 몰탈 자동 도포 |
| L1 | 성공률 70% | 팔레트 ±5 cm/±15° 랜덤화 |
| L2 | 성공률 70% | 몰탈 도포를 액션/스크립트로 실제 수행, 도포 품질이 결과에 반영 |
| L3 | 성공률 70% | 1단 4장 순차 배치(이웃 벽돌 존재) |
| L4 | 성공률 70% | **베이스 해제** — 자세 제어기 위에서 학습, 몸체 흔들림 발생 |
| L5 | 성공률 70% | 2~3단 확장(어긋쌓기, 몸체 자세 조절 필수 구간) |
| L6 | 성공률 70% | 도메인 랜덤화 풀가동 + 허용오차 강화(5 mm → 3 mm), 액션 스케일 축소 |

---

## 6. 씬(Scene) 구성

```
/World/envs/env_.*/
├── Robot            (Spot + Arm, ArticulationCfg, 초기 자세 = 표준 스탠스)
├── Ground           (콘크리트 슬래브, 마찰 0.9)
├── Pallet/Brick_src_[0..15]   (RigidObjectCollectionCfg, 스폰 랜덤화)
├── Wall/Brick_placed_[0..11]  (RigidObjectCollectionCfg, 미사용 시 지하 대기 후 텔레포트)
├── MortarNozzle     (EE 부착 고정 조인트, 트랙 A는 시각 전용)
├── ContactSensor    (finger ×2, 발 ×4, 벽돌 하면)
└── FrameTransformer (ee_frame ↔ brick ↔ slot 마커)
```

**물리 튜닝 — 초기 실패의 대부분이 여기서 발생합니다.**

| 파라미터 | 값 | 이유 |
|---|---|---|
| `dt` | 1/120 | 스택 태스크에서 큰 dt는 즉시 붕괴 |
| `solver_position_iteration_count` | 16~24 | 다층 스택 안정성 |
| `solver_velocity_iteration_count` | 1 | 위치 반복 우선 |
| 벽돌 마찰 | static 0.9 / dynamic 0.8, restitution 0.0 | 몰탈 접촉 근사 |
| `contact_offset` / `rest_offset` | 0.002 / 0.0005 | 과도한 offset은 "떠 있는 벽" 유발 |
| `max_depenetration_velocity` | 1.0 | 관통 복구 시 폭발 방지 |
| 충돌 형상 | **box collider** (메시 아님) | 성능·안정성 모두 유리 |
| CCD | off | 저속 조작 태스크에 불필요, 비용만 큼 |

---

## 7. 리포지토리 구조

Isaac Lab 본체를 포크하지 않고 **외부 확장(external extension)** 으로 갑니다 (`./isaaclab.sh --new`).

```
masonry_rl/
├── source/masonry_rl/masonry_rl/
│   ├── tasks/masonry/
│   │   ├── __init__.py                 # Isaac-Masonry-Wall-SpotArm-IK-Rel-v0
│   │   ├── masonry_env_cfg.py          # ManagerBasedRLEnvCfg
│   │   ├── spot_arm_cfg.py             # Spot+Arm ArticulationCfg, 액추에이터
│   │   ├── mdp/
│   │   │   ├── observations.py         # 슬롯 잔여오차, 몰탈 두께, 몸체 상태
│   │   │   ├── rewards.py              # align/seat_force/joint_thickness/release_stable
│   │   │   ├── terminations.py         # 붕괴·전도·과대력
│   │   │   ├── events.py               # 리셋 랜덤화
│   │   │   ├── actions.py              # 몸체 자세 액션 → 자세 제어기 브리지
│   │   │   └── curriculums.py
│   │   ├── mortar/
│   │   │   ├── reduced_model.py        # 트랙 A: 두께 필드 (torch 벡터화)
│   │   │   └── pbd_demo.py             # 트랙 B: 데모용 PBD 입자
│   │   ├── wall_planner.py             # 슬롯 시퀀스 → 목표 pose
│   │   ├── stance_controller.py        # 제자리 자세 유지(P4)
│   │   └── agents/rsl_rl_ppo_cfg.py
│   └── assets/                         # brick.usd, pallet.usd, nozzle.usd
├── scripts/
│   ├── train.py / play.py
│   ├── test_stack_stability.py         # P1 게이트 검증
│   ├── eval_wall.py                    # 벽 완성도 배치 평가
│   └── record_demo.py                  # 트랙 B 렌더 + 영상 캡처
├── configs/
└── docs/
```

---

## 8. 학습 설정 — RTX A6000 (48 GB) 기준

### 8.1 병렬 환경 수

| 단계 | `num_envs` | 근거 |
|---|---|---|
| P1~P2 (디버깅, GUI) | **32** | 시각 확인용 |
| P3 (베이스 고정, 팔만) | **2048** | 환경당 강체 ~20개 + 접촉센서. 48 GB에서 여유 |
| P4~P5 (다리 활성 + 몰탈 + 3단) | **1024** | 관절 18개 + 접촉쌍 급증. VRAM보다 **접촉 계산**이 병목 |
| P6 (데모 렌더) | **1~4** | RTX 렌더 + PBD 입자 |

> RTX PRO 6000 Blackwell(96 GB)이라면 각각 4096 / 2048로 상향.

### 8.2 PhysX GPU 버퍼 — **필수 조정**

벽돌 스택 + 4족 접촉 조합은 기본 버퍼를 넘겨 `PxgDynamicsMemoryConfig` 오버플로로 죽습니다. 학습 시작 전 반드시 상향하십시오.

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
num_envs          : 2048 (P3) / 1024 (P4+)
num_steps_per_env : 32
mini_batches      : 4
learning_epochs   : 5
learning_rate     : 1.0e-3, adaptive (desired_kl = 0.01)
gamma             : 0.99      lambda : 0.95
entropy_coef      : 0.005     clip   : 0.2
max_grad_norm     : 1.0       value_loss_coef : 1.0
network           : MLP [512, 256, 128], ELU, 비대칭 actor-critic
empirical_normalization : True
max_iterations    : 4000
```

관측 차원이 크고(≈95) 부유 베이스 태스크라 네트워크를 [256,128,64]에서 **[512,256,128]로 키웠습니다.**

### 8.4 예상 학습 시간 (A6000 단일)

| 단계 | 예상 처리량 | 총 스텝 | 예상 wall-clock |
|---|---|---|---|
| P3 (팔만, 2048 env) | 15~25k FPS | 약 2.0e8 | **3~4 시간** |
| P4 (다리+몰탈, 1024 env) | 6~10k FPS | 약 2.5e8 | **7~11 시간** |
| P5 (3단 + DR, 1024 env) | 5~8k FPS | 약 3.0e8 | **10~16 시간** |

야간 1회 실행으로 각 단계가 끝나는 규모입니다. 학습은 반드시 `--headless`(렌더 off), 영상은 `play.py --enable_cameras`로 별도 실행.

### 8.5 운영 팁

- 로깅: TensorBoard 기본 + W&B 선택. 커리큘럼 레벨 분포를 반드시 로깅(레벨이 안 오르면 조기에 발견해야 함).
- 체크포인트: 200 iteration마다 저장, 성공률 기준 best 별도 보관.
- A6000은 fp32로 충분(PhysX 자체가 fp32). AMP 도입 이득 미미하므로 생략.
- 학습 중 다른 GUI 세션을 띄우지 말 것 — VRAM 경합으로 접촉 버퍼가 터집니다.

---

## 9. 로드맵

| Phase | 목표 | 완료 게이트 | 예상 |
|---|---|---|---|
| **P0. 자산 확보** | Isaac Lab 설치·버전 고정, **Spot Arm 자산 3순위 전략 실행**, 레퍼런스 태스크 재현 | Spot이 서서 팔로 바닥 물체 파지 성공 | 3~5일 |
| **P1. 씬 & 물리** | 벽돌/팔레트/노즐 USD, 씬 cfg, 물리·GPU 버퍼 튜닝 | 3단 벽 수동 배치 후 10초 무붕괴, 1024 env 버퍼 오버플로 없음 | 3~4일 |
| **P2. MDP 구현** | obs/action/reward/termination + 축약 몰탈 모델 | 랜덤 정책 구동, 보상 신호 sanity check, 몰탈 두께 필드 검증 | 4~6일 |
| **P3. 팔 단독 학습** | 베이스 고정, L0~L3 | 1단 4장 성공률 ≥ 85% | 5~7일 |
| **P4. 베이스 해제** | 자세 제어기 + 몸체 자세 액션, L4 | 몸체 흔들림 하 성공률 ≥ 80%, 전도 0% | 1~2주 |
| **P5. 3단 벽 + 강건화** | L5~L6, 몰탈 도포 학습 승격, 도메인 랜덤화 | 벽 완성률 ≥ 70%, 줄눈 유지율 ≥ 80% | 2~3주 |
| **P6. 데모 산출** | 트랙 B PBD 몰탈, 카메라 리그, 렌더 영상 | 1~4 env 고품질 영상 + 평가 리포트 | 4~6일 |

**총 예상: 6~9주** (P0 자산 확보 결과에 따라 변동)

---

## 10. 평가 지표

| 지표 | 정의 |
|---|---|
| Placement success rate | 허용오차 내 착좌 + 안정 유지 비율 |
| Position / Yaw RMSE | 목표 슬롯 대비 최종 오차 |
| **Joint thickness distribution** | 줄눈 두께 분포 (8~12 mm 유지율) |
| **Course levelness** | 단별 상면 수평도 누적 오차 |
| Collapse rate | 기 배치 벽돌 붕괴 비율 |
| **Body sway RMS** | 착좌 구간 몸체 롤·피치 변동 (Spot 특화 지표) |
| Peak seating force | 착좌 시 최대 수직력 |
| Cycle time | 벽돌 1장당 소요 시뮬 시간 |
| Wall completion | 12장 중 성공 배치 비율 |

`eval_wall.py`로 100 에피소드 배치 평가 + 실패 케이스 영상 자동 저장.

---

## 11. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **Spot Arm 자산 부재** | **치명적 · 최우선** | §2의 3단계 폴백. P0 게이트 통과 전 진행 금지 |
| **부유 베이스 때문에 mm 정밀도 미달** | 높음 | ① 베이스 고정 학습 선행(P3) ② 착좌 마지막 2 cm는 impedance 기반 스크립트 삽입 동작으로 위임하는 하이브리드 ③ 몸체 자세를 액션에 포함해 정책이 스스로 안정 자세를 찾게 함 |
| **몰탈 모델 과욕** | 높음 | 학습에는 축약 모델만. PBD 입자는 데모 전용으로 격리(§4). 이 경계를 넘지 말 것 |
| **희소 보상 탐색 실패** | 높음 | Isaac Lab Mimic으로 소수 텔레오퍼레이션 데모 수집 → 자동 증강 → BC 사전학습 후 PPO 파인튜닝 |
| **오차 누적으로 3단에서 붕괴** | 중간 | `joint_thickness`·`course levelness` 보상 강화, 하부 단 확정 후 fixed joint 고정 |
| **접촉 버퍼 오버플로 / 학습 속도 부족** | 중간 | §8.2 버퍼 상향, 활성 강체 수 상한(대기 벽돌은 sleep), box collider 고정 |
| **일정 지연** | 중간 | P3까지가 핵심 가치. P5·P6는 축소 가능(3단→2단, 영상 길이 단축) |

---

## 부록 A. 참고 Isaac Lab 레퍼런스

| 레퍼런스 | 재사용 포인트 |
|---|---|
| `Isaac-Velocity-Flat-Spot-v0` | Spot 액추에이터 게인, 다리 제어, 몸체 상태 관측 |
| `Isaac-Lift-Cube-Franka-v0` | 보상 shaping 구조, IK 액션 골격 |
| `Isaac-Stack-Cube-Franka-IK-Rel-v0` | 다중 물체 스택, 단계형 종료, 데모 수집 파이프라인 |
| Isaac Lab Mimic | 소수 데모 → 대량 데이터 생성 (탐색 실패 대응책) |

> API 시그니처는 설치 버전에 따라 다릅니다. P0에서 버전을 태그로 고정하고, 위 태스크 소스를 실제로 열어 현재 버전 API에 맞춰 본 문서의 클래스·필드명을 확정합니다.

## 부록 B. P0 착수 체크리스트

- [ ] Isaac Sim / Isaac Lab 설치, 버전 태그 고정, `docs/setup.md` 작성
- [ ] A6000 드라이버·CUDA 확인, `Isaac-Velocity-Flat-Spot-v0` 학습 재현(FPS 실측 → §8.4 추정치 보정)
- [ ] **Spot Arm 자산 확보** (1→2→3순위 순차 시도, 결과를 `docs/asset-notes.md`에 기록)
- [ ] `./isaaclab.sh --new` 확장 스캐폴딩 생성
- [ ] Spot이 선 자세로 바닥 물체 파지 — 스크립트 데모 성공 (**P0 게이트**)
