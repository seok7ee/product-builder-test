# Isaac Lab 조적(Masonry / Bricklaying) 강화학습 프로젝트 계획서

> 상태: 초안 v0.1 · 대상: Isaac Lab 2.x (Isaac Sim 4.5+) · 작성 목적: 구현 착수 전 설계 합의

---

## 0. 한 줄 정의

**로봇 매니퓰레이터가 벽돌 팔레트에서 벽돌을 집어 → 이동 → 목표 슬롯에 정밀 정렬 → 압착 배치 → 놓기**를 반복해 벽(course, 단)을 쌓아 올리는 정책을, Isaac Lab의 GPU 병렬 시뮬레이션에서 PPO로 학습한다.

성공 기준(최종): 4096 병렬 환경에서 학습한 단일 정책이 **1단 5장 이상을 연속 배치**하면서
- 위치 오차 ≤ 5 mm, yaw 오차 ≤ 2°
- 기 배치 벽돌 붕괴율 ≤ 5%
- 성공률 ≥ 85% (랜덤화된 팔레트/목표 위치에서)

---

## 1. 범위 결정 (기본값 + 대안)

| 항목 | 기본값 (권장) | 대안 | 결정 근거 |
|---|---|---|---|
| 로봇 | **Franka Panda (7-DoF + 병렬 그리퍼)** | UR10e+Robotiq, 모바일 매니퓰레이터(Ridgeback+Panda) | Isaac Lab 기본 자산·튜닝된 IK·기존 Lift/Stack 태스크 재사용 가능. 벽 전체(가로 이동)는 Phase 5에서 모바일 베이스로 확장 |
| 워크플로 | **Manager-based (`ManagerBasedRLEnv`)** | Direct (`DirectRLEnv`) | 보상/종료/커리큘럼을 term 단위로 켜고 끄며 실험하기 좋음. 단계형(reach→grasp→place) 태스크에 유리 |
| 액션 | **Task-space 상대 IK (`DifferentialInverseKinematicsAction`, 6D delta + gripper binary)** | 관절 위치, OSC(힘제어) | 조적은 위치 정밀도 태스크. 관절공간 직접 학습은 샘플 효율이 매우 나쁨. 압착 단계만 Phase 4에서 OSC로 교체 검토 |
| RL 라이브러리 | **rsl_rl (PPO)** | skrl(SAC/PPO), rl_games | Isaac Lab 레퍼런스 태스크와 하이퍼파라미터 정합성이 가장 높음. skrl은 알고리즘 비교 실험 때 추가 |
| 몰탈(모르타르) | **Phase 1~3 생략, Phase 4에서 근사 모델** | 입자/유체 시뮬 | 실제 몰탈 시뮬은 비용이 과도. "접촉면 고마찰 + 배치 후 fixed joint 생성 + 압착력 목표치" 로 근사 |
| 벽돌 | **표준 규격 190×90×57 mm 강체(rigid body)** | 실제 무게/치수 변동 포함 | 초기엔 균일, 도메인 랜덤화 단계에서 ±5% 치수/질량 변동 |

> ⚠️ 위 표는 **가정**입니다. 로봇 하드웨어가 이미 정해져 있거나(예: 현장 로봇이 UR/두산/현대로보틱스), 모바일 베이스가 필수라면 Phase 0에서 말씀 주시면 자산/액션 설계를 그에 맞춰 교체합니다.

---

## 2. 태스크를 왜 쪼개야 하는가

조적은 **long-horizon + contact-rich + 희소 보상** 3중고 태스크입니다. 벽 한 줄을 통째로 end-to-end 학습하면 거의 확실히 실패합니다. 그래서:

1. **단위 태스크 = 벽돌 1장 배치 (pick-and-place-one-brick)** 로 정의하고,
2. 벽은 "이 단위 태스크의 반복 + 목표 슬롯 이동"으로 구성하며,
3. 이미 쌓인 벽돌은 **관측/충돌 대상이자 붕괴 페널티 소스**로만 존재하게 한다.

즉 **정책은 "벽을 쌓는 법"이 아니라 "주어진 슬롯에 벽돌 1장을 안전하게 놓는 법"을 배운다.** 슬롯 순서 결정은 학습이 아닌 결정론적 플래너(단→열 순서)가 담당합니다. 이 분리가 이 프로젝트의 핵심 설계 결정입니다.

---

## 3. MDP 설계

### 3.1 관측 (policy obs, ~60~80 dim)

| 그룹 | 항목 | 차원 |
|---|---|---|
| 고유수용 | 관절 위치(상대), 관절 속도, 그리퍼 개폐 상태 | 7+7+1 |
| EE | EE pose (pos + 6D rot), EE 선/각속도 | 9+6 |
| 대상 벽돌 | EE 기준 상대 위치, 상대 자세(6D), 파지 여부 플래그 | 9+1 |
| 목표 슬롯 | EE 기준 상대 위치, 벽돌 기준 상대 위치·자세(= 잔여 오차) | 3+9 |
| 벽 상태 | 현재 단 번호, 슬롯 인덱스 정규화값, 직전 배치 벽돌 상대 위치 | 5 |
| 접촉 | 그리퍼 finger contact force (clip), EE 수직 접촉력 | 3 |
| 이전 액션 | last action | 7 |

- **critic obs(비대칭 actor-critic)**: 위 + 특권 정보(벽돌 절대 pose, 마찰계수, 질량, 모든 배치 벽돌의 변위) → 학습 안정화. Isaac Lab의 `ObservationGroupCfg`로 `policy` / `critic` 분리.
- 노이즈: 위치 ±2 mm, 각도 ±1° 가우시안 (sim2real 대비).

### 3.2 액션 (7 dim)

- `[dx, dy, dz, drx, dry, drz]` EE 상대 delta (스케일: 위치 0.05 m, 회전 0.05 rad) + `[gripper]` binary
- 정밀 배치 구간에서는 액션 스케일을 커리큘럼으로 축소(0.05 → 0.01)해 미세 조정 능력 확보

### 3.3 보상 (staged shaping)

```
r = w1·reach + w2·grasp + w3·lift + w4·align + w5·place + w6·release_stable
    - p1·collapse - p2·excess_force - p3·action_rate - p4·joint_vel - p5·collision
```

| Term | 정의 | 가중치(초기) |
|---|---|---|
| `reach` | `1 - tanh(d_ee_brick / 0.1)` | 1.0 |
| `grasp` | 양쪽 finger 접촉 + 그리퍼 닫힘 시 상수 | 2.0 |
| `lift` | 벽돌 높이 > 임계(3 cm) 시 상수 | 5.0 |
| `align` | 파지 상태에서 `1 - tanh(d_brick_slot / 0.15)` + yaw 정렬 `cos(Δyaw)` | 8.0 |
| `place` | 슬롯 허용오차(5 mm / 2°) 내 진입 시 큰 보너스 | 20.0 |
| `release_stable` | 그리퍼 개방 후 N=30 step 동안 벽돌 변위 < 3 mm | 30.0 |
| `collapse` | 기 배치 벽돌 중 하나라도 변위 > 1 cm | -50.0 |
| `excess_force` | EE 수직 접촉력 > 50 N 초과분 | -0.1/N |
| `action_rate`, `joint_vel` | 표준 정규화 페널티 | -0.01 |

> 설계 원칙: **`release_stable`이 최종 보상의 주축.** 놓는 순간이 아니라 "놓고 나서도 안 무너지는가"로 성공을 정의해야 실제 조적 품질과 일치합니다.

### 3.4 종료 조건

- 성공: `release_stable` 달성 → 다음 벽돌로 리셋(에피소드 내 연속 배치 모드에서는 슬롯만 전진)
- 실패: 벽돌 낙하(z < 테이블면 - 5 cm), 붕괴 감지, 접촉력 폭주(> 150 N), 로봇 자기충돌
- 타임아웃: 단일 벽돌 8초 (dt=1/120, decimation=4 → 240 step)

### 3.5 커리큘럼

| 레벨 | 조건 | 변화 |
|---|---|---|
| L0 | 시작 | 벽돌 픽업 위치 고정, 목표 슬롯 1개, 빈 벽 |
| L1 | 성공률 70% | 픽업 위치 ±5 cm/±15° 랜덤화 |
| L2 | 성공률 70% | 1단 내 슬롯 5개 순차 배치(이웃 벽돌 존재) |
| L3 | 성공률 70% | 2~3단 (아래 단 위 배치 = 접촉 정밀도 상승) |
| L4 | 성공률 70% | 도메인 랜덤화 풀가동(마찰 0.4~1.0, 질량 ±10%, 치수 ±3 mm, 관측 노이즈 증가) |
| L5 | 성공률 70% | 액션 스케일 축소 + 허용오차 강화(5 mm → 3 mm) |

Isaac Lab `CurriculumTermCfg`로 구현, 환경별 독립 레벨(per-env difficulty) 권장.

---

## 4. 씬(Scene) 구성

```
/World/envs/env_.*/
├── Robot          (Franka Panda, ArticulationCfg)
├── Table          (고정 강체)
├── Pallet/Brick_source_[0..N]   (RigidObjectCollectionCfg, 스폰 랜덤화)
├── Wall/Brick_placed_[0..M]     (RigidObjectCollectionCfg, 초기 비활성/지하 배치 후 필요 시 텔레포트)
├── ContactSensor  (finger_left, finger_right, ee_link)
└── FrameTransformer (ee_frame ↔ brick ↔ slot 마커)
```

핵심 물리 튜닝 포인트(여기서 대부분의 초기 실패가 발생):
- `solver_position_iteration_count` ≥ 16 (스택 안정성)
- 벽돌 static/dynamic friction 0.8 / 0.7, restitution 0.0
- `contact_offset` 0.002 / `rest_offset` 0.0005 (벽돌 두께 대비 과도한 offset은 "떠 있는 벽" 유발)
- 그리퍼 finger에 높은 마찰 머티리얼 + `max_depenetration_velocity` 제한
- CCD는 끄고, dt=1/120 유지 (스택 태스크에서 큰 dt는 즉시 붕괴)

---

## 5. 리포지토리 구조

Isaac Lab 본체를 포크하지 않고 **외부 확장(external extension) 템플릿** 방식으로 갑니다 (`./isaaclab.sh --new`).

```
masonry_rl/
├── source/masonry_rl/masonry_rl/
│   ├── tasks/masonry/
│   │   ├── __init__.py                 # gym 등록: Isaac-Masonry-Brick-Franka-IK-Rel-v0
│   │   ├── masonry_env_cfg.py          # ManagerBasedRLEnvCfg (scene/obs/rew/term/curriculum)
│   │   ├── mdp/
│   │   │   ├── observations.py         # 슬롯 상대오차, 벽 상태 관측
│   │   │   ├── rewards.py              # align/place/release_stable/collapse
│   │   │   ├── terminations.py         # 붕괴·낙하·과대력
│   │   │   ├── events.py               # 리셋 랜덤화, 슬롯 플래너
│   │   │   └── curriculums.py
│   │   ├── wall_planner.py             # 슬롯 시퀀스 생성(단/열 → 목표 pose)
│   │   └── agents/rsl_rl_ppo_cfg.py
│   └── assets/                         # brick.usd, pallet.usd, 로봇 cfg 오버라이드
├── scripts/
│   ├── train.py / play.py              # 템플릿 제공본 사용
│   ├── record_demo.py                  # (Phase 4) 텔레오퍼레이션 데모 수집
│   └── eval_wall.py                    # 벽 완성도 평가 하니스
├── docs/
└── configs/
```

---

## 6. 단계별 로드맵

| Phase | 목표 | 산출물 | 완료 기준 | 예상 |
|---|---|---|---|---|
| **P0. 환경 셋업** | Isaac Sim/Lab 설치, GPU 확인, 레퍼런스 태스크 구동 | 설치 문서, `Isaac-Lift-Cube-Franka-v0` 학습 재현 | 레퍼런스 태스크 성공률 재현 | 1~2일 |
| **P1. 씬 & 자산** | 벽돌/팔레트 USD, 씬 cfg, 물리 튜닝 | `masonry_env_cfg.py` 씬 파트, 스택 안정성 테스트 스크립트 | 벽돌 3단 수동 배치 후 10초간 무붕괴 | 2~3일 |
| **P2. MDP 구현** | obs/action/reward/termination 전체 | `mdp/*.py`, gym 등록 | `play.py`로 랜덤 정책 구동, 보상 신호 sanity check | 3~4일 |
| **P3. 단일 벽돌 학습** | L0~L1 커리큘럼 PPO 학습 | 학습 로그(W&B/TensorBoard), 체크포인트 | 고정 슬롯 성공률 ≥ 90% | 3~5일 |
| **P4. 1단 벽 + 몰탈 근사** | L2~L3, 압착(force) 단계 추가 | OSC 액션 옵션, 몰탈 근사 모델 | 5장 연속 배치 성공률 ≥ 85%, 붕괴 ≤ 5% | 1~2주 |
| **P5. 확장/강건화** | L4~L5 도메인 랜덤화, 다단 벽, (선택) 모바일 베이스 | 평가 리포트, 정책 비교표 | 3단 × 5열 벽 완성률 ≥ 70% | 2~3주 |
| **P6. (선택) Sim2Real 준비** | 관측 노이즈/지연 모델, ROS 2 브리지 | 정책 export(ONNX/TorchScript), 인터페이스 문서 | 실기 없이 HIL 수준 검증 | 별도 산정 |

---

## 7. 학습 설정 (초기값)

```
알고리즘: PPO (rsl_rl)
num_envs: 4096  (P1~P3는 디버깅 위해 64, headless 학습 시 4096)
rollout: 24 steps  |  minibatch: 4  |  epochs: 5
lr: 1e-3 adaptive (desired_kl=0.01)  |  gamma: 0.99  |  lam: 0.95
entropy_coef: 0.006  |  clip: 0.2  |  max_grad_norm: 1.0
network: MLP [256,128,64], ELU, 비대칭 actor-critic
max_iterations: 3000 (~1.5억 step)
```

권장 하드웨어: RTX 4090 / A6000 이상, VRAM 24 GB+. 4096 env 조적 태스크는 접촉 계산이 무거워 큐브 태스크 대비 **2~3배 느립니다**(체감 20k~40k FPS → 8k~15k FPS).

---

## 8. 평가 지표

| 지표 | 정의 |
|---|---|
| Placement success rate | 허용오차 내 배치 + 안정 유지 비율 |
| Position/Yaw RMSE | 목표 슬롯 대비 최종 오차 |
| Collapse rate | 에피소드 중 기 배치 벽돌 붕괴 비율 |
| Cycle time | 벽돌 1장당 소요 시뮬 시간 |
| Wall completion | 목표 벽 대비 완성 벽돌 수 비율 |
| Peak contact force | 배치 시 최대 수직력(과압착 여부) |

`eval_wall.py`에서 100 에피소드 배치 평가 + 실패 케이스 영상 자동 저장.

---

## 9. 주요 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **정밀 배치가 RL만으로 안 됨** (mm 단위 정렬은 PPO 난제) | 치명적 | 하이브리드: RL은 접근·정렬까지, 마지막 2 cm는 **impedance/admittance 기반 스크립트 삽입 동작**으로 위임. P3에서 순수 RL 성공률 미달 시 즉시 전환 |
| **희소 보상으로 탐색 실패** | 높음 | 데모 기반 부트스트랩: Isaac Lab Mimic으로 소수 텔레오퍼레이션 데모 → 자동 증강 → BC 사전학습 후 PPO 파인튜닝 |
| **스택 물리 불안정 (시뮬 아티팩트)** | 높음 | P1에서 물리 파라미터를 먼저 확정. 필요 시 배치 확정 벽돌은 `kinematic` 전환 또는 fixed joint로 고정 |
| **학습 속도 부족** | 중간 | 벽 벽돌 수 제한(활성 강체 상한), 관측 단순화, LOD 낮은 충돌 메시(box collider 사용) |
| **몰탈 모델링 과욕** | 중간 | Phase 4까지 유체/입자 시뮬은 금지. 접촉 파라미터 근사로 한정 |
| **Sim2Real 갭** | 후반 | 도메인 랜덤화 + 액추에이터 지연 모델 + 관측 노이즈를 P5에 필수 포함 |

---

## 10. 즉시 확인이 필요한 사항

구현 착수 전 다음이 정해지면 계획을 바로 확정본으로 바꿀 수 있습니다.

1. **로봇 기종** — Franka로 프로토타이핑 후 교체할지, 처음부터 특정 상용 로봇(UR/두산 등) URDF로 갈지
2. **작업 규모** — 테이블 위 소형 벽(1×5×3) 데모인지, 실제 건축 스케일(모바일 베이스 + 벽 길이 수 m)인지
3. **몰탈 포함 여부** — 무몰탈 건식 조적(interlocking block)이면 난이도가 크게 낮아짐
4. **최종 목적** — 논문/데모용 시뮬 결과까지인지, 실기 이식(sim2real)까지인지
5. **하드웨어** — 사용 가능한 GPU (학습 시간 산정 및 num_envs 결정)

---

## 부록 A. 참고할 Isaac Lab 레퍼런스 태스크

| 태스크 | 재사용 포인트 |
|---|---|
| `Isaac-Lift-Cube-Franka-v0` | 보상 shaping 구조, IK 액션 설정의 기본 골격 |
| `Isaac-Stack-Cube-Franka-IK-Rel-v0` | 다중 물체 스택, 단계형 종료 조건, 데모 수집 파이프라인 |
| `Isaac-Open-Drawer-Franka-v0` | 접촉 기반 조작, FrameTransformer 사용 예 |
| Isaac Lab Mimic | 소수 데모 → 대량 데이터 자동 생성 (P4 리스크 대응책) |

> API 시그니처는 설치된 Isaac Lab 버전에 따라 다릅니다. P0에서 버전을 고정(`git tag`)하고, 위 레퍼런스 태스크 소스를 실제로 열어 현재 버전 API에 맞춰 계획서의 클래스명을 확정합니다.

## 부록 B. P1 착수 시 첫 커밋 체크리스트

- [ ] Isaac Lab 버전 고정 및 `docs/setup.md` 작성
- [ ] `./isaaclab.sh --new` 로 확장 스캐폴딩 생성
- [ ] `brick.usd` 생성(190×90×57 mm, box collider, 마찰 머티리얼)
- [ ] 씬 cfg + `scripts/test_stack_stability.py` (수동 3단 배치 후 붕괴 여부 측정)
- [ ] gym 등록 및 랜덤 액션 구동 확인
