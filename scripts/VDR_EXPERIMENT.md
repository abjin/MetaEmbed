# VDR-only MetaEmbed 실험 (query 16 고정 + doc MMR)

MetaEmbed를 문서 이미지 검색(VDR) 데이터만으로 학습하고, 쿼리 Meta Token 수는 16으로 고정한 채
문서 쪽 토큰 예산만 MMR로 학습했을 때 ViDoRe 성능이 토큰 수에 따라 어떻게 바뀌는지 본다.

## 설정

| 항목 | 값 |
|---|---|
| 학습 데이터 | MoCa-CL-Pairs `tevatron_colpali` 118,194 + `visrag_ind` 122,752 (weight 5:5), hard negative 1개 |
| MMR 그룹 | `[16,1] [16,4] [16,8] [16,16] [16,64]` (쿼리 16 고정, 문서 1/4/8/16/64) |
| 모델 | Qwen2.5-VL-3B / 7B-Instruct + processor `vidore/colqwen2.5-base`, Meta Token 16/64, dim -1 |
| LoRA | r 32, alpha 32, dropout 0.1, LLM의 attention/MLP proj만 (transformers 4.51.3 기준 7B 81.0M = 0.97%) |
| 최적화 | lr 1e-4, warmup 40 step, τ 0.03, `do_gather: true` (전 GPU의 positive를 negative로 사용) |
| 쿼리당 후보 | 321개 = 전체 배치의 positive 320 + 자기 hard negative 1 (hard negative는 gather하지 않음, `contrastive_trainer_v2.py:184`) |
| 배치 | 전체 320, 1 epoch = 750 step. 3B: GPU 4장(4–7번) × 80 / 7B: GPU 5장(3–7번) × 64 |
| 평가 | 150 step마다 ViDoRe v1 10개(`mteb/*_beir`) + v2 4개(`vidore/*_v2`), MMEB 끔 |

논문(배치 1,536, MMEB+ViDoRe 데이터, 약 4.5 epoch)과 다른 점은 의도된 것이다.

## 파일

- `scripts/7b_mrl_vdr.yaml`, `scripts/3b_mrl_vdr.yaml`: 학습 설정 (베이스 모델, 출력 폴더, 장당 배치만 다름)
- `scripts/3b_mrl_vdr_mixed.yaml`: 3B + 혼합 토큰 수 로스 (아래 "혼합 토큰 수 로스")
- `scripts/3b_mrl_vdr_evalmixed.yaml` + `scripts/eval/eval_checkpoints.py`: 학습이 끝난 체크포인트를 평가만 다시 돌림
- `scripts/vdr_data_ratios.yaml`: 학습 subset과 비율
- `scripts/data/prepare_moca_vdr.py`: MoCa parquet → 학습 로더 형식 변환
- `scripts/data/test_data_vidore_beir_hf.yaml`, `scripts/data/test_data_vidore_v2_hf.yaml`: HF id 기반 평가셋
- `fsdp_4gpu.yaml`, `fsdp_5gpu.yaml`: accelerate FSDP2 설정 (`num_processes`만 다름)
- `scripts/train/run_vdr.sh`: 학습 실행 스크립트
- `requirements-vdr-lock.txt`: 실제로 돌린 환경의 `uv pip freeze`

## 재현 순서

```bash
# 1) 환경 (Python 3.11, uv). torch는 cu128 index에서 설치된다.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-vdr-lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

# 2) 데이터: HF moca-embed/MoCa-CL-Pairs 의 두 subset parquet와
#    supplement_images_zip/*.tar.gz 를 <raw>/ 에 받고, tar 는 <raw>/images/ 에 푼다.
.venv/bin/python scripts/data/prepare_moca_vdr.py \
  --raw-dir /home/gijin/sample-adaptive-embedding/data_raw/moca \
  --out-dir /home/gijin/sample-adaptive-embedding/data_dir
#    이미지 컬럼은 절대경로만 저장하므로 raw 폴더 위치를 바꾸면 다시 변환해야 한다.

# 3) 학습 (로그는 ../runs/<config>.log, 체크포인트와 평가 json은 scripts/output/<output_dir>/)
GPUS=4,5,6,7 FSDP_CONFIG=fsdp_4gpu.yaml scripts/train/run_vdr.sh scripts/3b_mrl_vdr.yaml
#    기본값은 GPU 3–7 + fsdp_5gpu.yaml. num_shards는 GPU 수 × 4(worker)로 나누어떨어져야 한다.
```

## 코드 수정 (원본 대비)

- `vidore_benchmark/**/__init__.py`, `vidore_evaluator_{beir,qa}.py`: 레포에 없는 retriever(bm25, bge_m3, cohere 등)와
  `EvalManager` import 때문에 trainer import가 깨져서, 있는 모듈만 import 하도록 최소 수정.
- `requirements.txt`: `mteb` 중복 항목(1.16.5) 제거. 1.38.12를 쓴다.

## 주의할 점

- **transformers는 4.51.3이어야 한다.** README는 4.55.0을 안내하지만, `modeling_lastqwen2_5.py`는 4.51.3 기준이다.
  4.55에서는 liger 패치가 `'Qwen2_5_VLModel' object has no attribute 'norm'`으로 실패하고,
  vision tower가 `model.visual` 아래로 옮겨져 LoRA 정규식에 같이 걸린다.
- `num_shards`(160)는 GPU 수 × dataloader worker 수(5×4)로 나누어떨어져야 한다.
  안 그러면 `split_dataset_by_node`가 샤드 분할 대신 예제 건너뛰기로 동작해 GPU 간 데이터가 겹친다.
- `vidore_eval_frequency`는 `eval_steps`의 배수여야 한다.
- MoCa 원본 tar에 0바이트 이미지가 1장 있다(`colpali_tevatron/colpali_96.jpg`). 변환 스크립트가 PIL로 열리지 않는 이미지가 든 행을
  버리므로(`tevatron_colpali` 1행) 변환 결과를 그대로 쓰면 된다. 예전 변환본을 쓰면 학습 중 DataLoader가 죽는다.
- **데이터를 다시 변환했으면 `~/.cache/huggingface/datasets/mo_ca_train_with_image`를 지워야 한다.** 안 지우면 `load_dataset`이
  예전 arrow 캐시를 그대로 읽는다. 로그의 `total num rows=240746 with [118094, 122652]`로 확인한다(학습용, subset당 평가용 100행 제외).
- 장당 64에서 7B는 GPU 메모리 최대 75.6GB/80GB(평가 포함)까지 쓴다. `expandable_segments`를 켜고 돌린다.
- **rpower4에서는 GPU 3번을 쓰지 않는다.** `nvidia-smi topo -m` 기준 0–3번과 4–7번이 서로 다른 PCIe 스위치·CPU 소켓(SYS)에 있고,
  3번은 다른 사용자가 쓰는 0–2번과 스위치를 나눠 쓴다. 3–7번 5장으로 돌린 두 번(7B, 3B) 모두 step당 시간이
  2–5배로 늘어나는 구간이 나왔고(GPU 사용률은 100%, 통신 대기로 추정), 4–7번 4장에서는 나오지 않았다.

## 혼합 토큰 수 로스 (`ColbertInBatchNegativeMixedBudgetLoss`)

기존 MMR은 그룹 하나 안에서 모든 후보 문서가 같은 토큰 수를 쓴다. 그래서 토큰 수가 서로 다른 문서끼리 점수를 비교할 수 있는지는
학습하지 않는다(문서마다 토큰 수를 다르게 색인하는 경우).
새 로스는 매 step **후보 문서마다** 토큰 수 r_c를 {1,4,8,16,64}에서 균등하게 독립적으로 뽑는다. positive와 hard negative 모두 해당한다.
쿼리 16토큰과 각 후보의 앞 r_c개 토큰으로 MaxSim 점수를 계산하고, 같은 321개 후보에 InfoNCE를 건다.

- 같은 문서는 모든 GPU에서 같은 r_c를 쓴다. gather된 positive는 GPU마다 같은 문서라서, (seed, 호출 횟수)로 시드한 생성기로 뽑는다. hard negative는 GPU별로 따로 뽑는다.
- r_c 뒤의 토큰은 -inf로 가려 max에서 뺀다. 0으로 채우면 음수 MaxSim이 0으로 바뀐다.
- 결합 방식(`mixed_budget_mode`)
  - `add`: MMR 평균 + w·L_mixed (기본, w=1.0)
  - `group`: (Σ MMR + w·L_mixed) / 6
- 로그 키는 `mixed_budget_16`이다.
- 평가에는 `mrl_16_mixed` 그룹이 추가된다. 문서 i마다 고정 seed(0)로 r_i를 뽑아 색인한 nDCG 등이다.
  - 모든 모델과 체크포인트가 같은 배정을 쓴다.
  - 스코어러가 0으로 패딩하는 문제는, 앞 r_i개 토큰 중 마지막 토큰을 반복해 길이를 맞추는 방식으로 피했다(MaxSim 값이 그대로다).
- 기준 모델에는 `scripts/eval/eval_checkpoints.py`로 같은 그룹을 사후 평가한다(GPU 1장).
  - **같은 체크포인트라도 학습 중 평가(GPU 4장)와 이 스크립트(GPU 1장)의 값이 다르다.** v1 평균 기준 스크립트 쪽이 0.02–0.03 꾸준히 높다.
    원인은 확인하지 못했다. GPU 수에 따라 배치 구성이 달라져 임베딩이 조금 바뀌는 것으로 추정한다.
  - 그래서 모델 간 비교는 반드시 같은 방식으로 잰 값끼리 한다. 아래 결과는 모두 이 스크립트로 잰 값이다.

### 결과: 기준 모델 vs + 혼합 토큰 수 로스 (3B, `eval_checkpoints.py`, nDCG@5 평균)

| step | 모델 | v1: 1 | 4 | 8 | 16 | 64 | 혼합 | v2: 1 | 4 | 8 | 16 | 64 | 혼합 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 | 기준 | 0.807 | 0.813 | 0.825 | 0.835 | 0.845 | 0.807 | 0.498 | 0.501 | 0.497 | 0.496 | 0.495 | 0.458 |
| 150 | +혼합 | 0.813 | 0.817 | 0.818 | 0.818 | 0.818 | 0.818 | 0.520 | 0.516 | 0.514 | 0.515 | 0.514 | 0.517 |
| 300 | 기준 | 0.840 | 0.845 | 0.860 | 0.870 | 0.877 | 0.845 | 0.546 | 0.541 | 0.534 | 0.545 | 0.542 | 0.521 |
| 300 | +혼합 | 0.837 | 0.845 | 0.849 | 0.849 | 0.851 | 0.848 | 0.565 | 0.557 | 0.553 | 0.553 | 0.553 | 0.554 |
| 450 | 기준 | 0.843 | 0.846 | 0.863 | 0.868 | 0.880 | 0.790 | 0.531 | 0.523 | 0.488 | 0.504 | 0.525 | 0.412 |
| 450 | +혼합 | 0.846 | 0.851 | 0.855 | 0.858 | 0.862 | 0.854 | 0.549 | 0.544 | 0.542 | 0.542 | 0.541 | 0.543 |
| 600 | 기준 | 0.851 | 0.854 | 0.867 | 0.872 | 0.878 | 0.747 | 0.526 | 0.520 | 0.496 | 0.511 | 0.520 | 0.373 |
| 600 | +혼합 | 0.855 | 0.865 | 0.865 | 0.867 | 0.869 | 0.864 | 0.541 | 0.535 | 0.535 | 0.534 | 0.537 | 0.537 |
| 750 | 기준 | 0.852 | 0.856 | 0.871 | 0.875 | **0.883** | 0.857 | 0.529 | 0.525 | 0.503 | 0.517 | 0.525 | 0.515 |
| 750 | +혼합 | **0.854** | **0.864** | 0.865 | 0.867 | 0.870 | **0.863** | **0.543** | **0.540** | **0.538** | **0.537** | **0.537** | **0.537** |

- **기준 모델은 혼합 색인에서 불안정하다.** 혼합 색인 성능이 체크포인트마다 크게 흔들린다(v1 0.747–0.857, v2 0.373–0.521).
  토큰 수가 다른 문서끼리 점수 크기가 맞지 않는다는 뜻이다.
- **혼합 로스를 넣으면 혼합 색인 성능이 고정 토큰 수 성능과 거의 같아지고, 모든 체크포인트에서 안정적이다.**
  step 750 혼합 색인은 v1 0.857 → 0.863, v2 0.515 → 0.537이다.
- **대가로 토큰 수에 따른 차이가 줄어든다.** v1에서 1토큰과 64토큰의 차이가 0.031에서 0.016으로 줄었다.
  - 1·4토큰은 좋아졌다.
  - 8토큰 이상은 나빠졌다. 64토큰은 0.883 → 0.870이고, 특히 tatdqa(0.761 → 0.727)와 shift_project(0.885 → 0.827)에서 크게 떨어졌다.
  - v2(다국어 3개 포함)는 모든 토큰 수에서 좋아졌다.
- 다음 후보: 혼합 로스 비중을 줄여 64토큰 손실을 줄이는지 본다(`group` 모드 또는 w=0.5).

## 실행 기록 (rpower4, A100 80GB)

| 날짜 | 실행 | 결과 |
|---|---|---|
| 2026-11-27 | 7B, GPU 4장 × 48, 50 step 테스트 (25 step에서 중단) | 37.3초/step, 메모리 59.8GB. step 25 tabfquad nDCG@5: 1토큰 0.673 / 4: 0.671 / 8: 0.665 / 16: 0.659 / 64: 0.655 |
| 2026-11-27 | 7B, GPU 4장 × 64, 20 step 테스트 | 48.3초/step, 메모리 75.6GB, 평가·체크포인트 저장 정상 |
| 2026-11-27 | 7B, GPU 5장 × 64, 본 학습 | 41/750 step에서 중단(3B 먼저 하기로 함). loss 10.98 → 2.25 (10 step 평균) |
| 2026-11-27 | 3B, GPU 5장 × 64, 20 step 테스트 | 31초/step(16–20 step은 120–140초로 느려짐), 메모리 41.4GB. step 20 tabfquad nDCG@5: 1토큰 0.567 / 4: 0.579 / 8: 0.590 / 16: 0.576 / 64: 0.544 |
| 2026-11-27 | 3B, GPU 4장(4–7) × 80, 본 학습 | 28 step에서 중단: 0바이트 이미지(`colpali_tevatron/colpali_96.jpg`) 디코딩 실패. 35.5초/step, 메모리 50.8GB |
| 2026-11-27 | 3B, 데이터 재생성 후 재시작 | 3 step에서 중단: HF datasets 캐시에 남은 예전 데이터를 읽고 있었음 |
| 2026-11-27 | 3B, GPU 4장(4–7) × 80, 본 학습 (캐시 삭제 후 재시작) | 완료(10시간 12분, 평가 포함). 학습 중 평가 step 750 v1 평균: 1토큰 0.824 / 64토큰 0.864. 평가 1회 약 30분 |
| 2026-11-27 | 3B + 혼합 토큰 수 로스, GPU 3번 1장 × 16, 5 step 테스트 | 정상. `mixed_budget_16`, `mrl_16_mixed` 확인 |
| 2026-11-27 | 기준 checkpoint-150, `eval_checkpoints.py` 테스트(tabfquad) | nDCG@5 1토큰 0.854 / 4: 0.853 / 8: 0.864 / 16: 0.884 / 64: 0.883 / 혼합 0.851 |
| 2026-11-27–28 | 3B + 혼합 토큰 수 로스 본 학습 (`add`, w=1.0) | 완료(학습 9시간 12분). 결과는 위 "결과" 표 |
| 2026-11-27–28 | 두 모델 체크포인트 5개씩 `eval_checkpoints.py` 평가 (GPU 3번) | 완료. 체크포인트당 약 30분 |

초반에는 MMR 그룹 5개의 loss가 거의 같게 움직였지만, step 150 평가에서는 토큰 수가 늘수록 nDCG가 꾸준히 올랐다(v1 평균 0.778 → 0.825).
Meta Token들이 서로 다른 정보를 나눠 담고 있다는 뜻으로 보인다. 다국어 v2에서는 차이가 작다.
