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
| 최적화 | lr 1e-4, warmup 40 step, τ 0.03, `do_gather: true` (전 GPU 배치를 negative로 사용) |
| 배치 | 전체 320, 1 epoch = 750 step. 3B: GPU 4장(4–7번) × 80 / 7B: GPU 5장(3–7번) × 64 |
| 평가 | 150 step마다 ViDoRe v1 10개(`mteb/*_beir`) + v2 4개(`vidore/*_v2`), MMEB 끔 |

논문(배치 1,536, MMEB+ViDoRe 데이터, 약 4.5 epoch)과 다른 점은 의도된 것이다.

## 파일

- `scripts/7b_mrl_vdr.yaml`, `scripts/3b_mrl_vdr.yaml`: 학습 설정 (베이스 모델, 출력 폴더, 장당 배치만 다름)
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

## 실행 기록 (rpower4, A100 80GB)

| 날짜 | 실행 | 결과 |
|---|---|---|
| 2026-11-27 | 7B, GPU 4장 × 48, 50 step 테스트 (25 step에서 중단) | 37.3초/step, 메모리 59.8GB. step 25 tabfquad nDCG@5: 1토큰 0.673 / 4: 0.671 / 8: 0.665 / 16: 0.659 / 64: 0.655 |
| 2026-11-27 | 7B, GPU 4장 × 64, 20 step 테스트 | 48.3초/step, 메모리 75.6GB, 평가·체크포인트 저장 정상 |
| 2026-11-27 | 7B, GPU 5장 × 64, 본 학습 | 41/750 step에서 중단(3B 먼저 하기로 함). loss 10.98 → 2.25 (10 step 평균) |
| 2026-11-27 | 3B, GPU 5장 × 64, 20 step 테스트 | 31초/step(16–20 step은 120–140초로 느려짐), 메모리 41.4GB. step 20 tabfquad nDCG@5: 1토큰 0.567 / 4: 0.579 / 8: 0.590 / 16: 0.576 / 64: 0.544 |
| 2026-11-27 | 3B, GPU 4장(4–7) × 80, 본 학습 | 28 step에서 중단: 0바이트 이미지(`colpali_tevatron/colpali_96.jpg`) 디코딩 실패. 35.5초/step, 메모리 50.8GB |
| 2026-11-27 | 3B, 데이터 재생성 후 재시작 | 3 step에서 중단: HF datasets 캐시에 남은 예전 데이터를 읽고 있었음 |
| 2026-11-27 | 3B, GPU 4장(4–7) × 80, 본 학습 (캐시 삭제 후 재시작) | 진행 중 |

초반에는 MMR 그룹 5개의 loss가 거의 같게 움직인다. 학습이 진행되면서 문서 64토큰 그룹이 1토큰 그룹보다
앞서는지(Meta Token이 실제로 다른 정보를 나눠 담는지)를 계속 확인해야 한다.
