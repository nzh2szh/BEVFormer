#!/usr/bin/env bash
set -euo pipefail

# Serial train->val runner for offline alignment in DDP mode.
# It trains exactly one new epoch per loop, then validates that epoch's align checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py}"
BASE_CKPT="${BASE_CKPT:-./ckpts/bevformer/epoch_24.pth}"

# Dataset profile: mini | trainval
DATASET_PROFILE="${DATASET_PROFILE:-mini}"
if [[ "${DATASET_PROFILE}" != "mini" && "${DATASET_PROFILE}" != "trainval" ]]; then
  echo "[ERROR] DATASET_PROFILE must be mini or trainval, got: ${DATASET_PROFILE}"
  exit 1
fi

if [[ "${DATASET_PROFILE}" == "mini" ]]; then
  DEFAULT_WORK_DIR="work_dirs/mini_ddp_offline_train_val"
  DEFAULT_SCENE_JSON="data/nuscenes/v1.0-mini/scene.json"
else
  DEFAULT_WORK_DIR="work_dirs/trainval_ddp_offline_train_val"
  DEFAULT_SCENE_JSON="data/nuscenes/v1.0-trainval/scene.json"
fi

WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"

# Epoch range: inclusive.
START_EPOCH="${START_EPOCH:-1}"
END_EPOCH="${END_EPOCH:-1}"
# Planned final epoch for this serial run. Note: the per-epoch loop still
# advances training by setting runner.max_epochs to the current epoch, so the
# train command returns after exactly one new epoch.
SERIAL_TOTAL_EPOCHS="${SERIAL_TOTAL_EPOCHS:-${END_EPOCH}}"

# DDP settings.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
TORCHRUN_RDZV_TIMEOUT="${TORCHRUN_RDZV_TIMEOUT:-7200}"
RDZV_ID="${RDZV_ID:-bevformer_align_ddp}"
TORCHRUN_RDZV_CONF_EXTRA="${TORCHRUN_RDZV_CONF_EXTRA:-}"
TORCHRUN_BACKEND_MODE="${TORCHRUN_BACKEND_MODE:-c10d}"

# Network interface binding for stable multi-node rendezvous/NCCL.
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
REQUIRE_SOCKET_IFNAME="${REQUIRE_SOCKET_IFNAME:-true}"

# Optional rsync-daemon checkpoint sync (for non-shared WORK_DIR multi-node runs).
# Only required resume checkpoints are synced: epoch_${epoch}.pth
ENABLE_RSYNC_SYNC="${ENABLE_RSYNC_SYNC:-false}"
RSYNC_PASSWORD_FILE="${RSYNC_PASSWORD_FILE:-/etc/rsync.password}"
RSYNC_TARGET_IP="${RSYNC_TARGET_IP:-192.168.103.3}"
RSYNC_TARGET_USER="${RSYNC_TARGET_USER:-rsync_user}"
RSYNC_TARGET_PATH="${RSYNC_TARGET_PATH:-backup_module/}"
RSYNC_EXTRA_OPTS="${RSYNC_EXTRA_OPTS:--avz}"
RESUME_WAIT_SECONDS="${RESUME_WAIT_SECONDS:-3600}"
RESUME_WAIT_INTERVAL="${RESUME_WAIT_INTERVAL:-10}"

# Memory allocator setting for PyTorch CUDA.
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

# Offline-train / offline-val settings.
SCENE_JSON="${SCENE_JSON:-${DEFAULT_SCENE_JSON}}"
TRAIN_ANN="${TRAIN_ANN:-data/nuscenes/nuscenes_infos_temporal_train.pkl}"
VAL_ANN="${VAL_ANN:-data/nuscenes/nuscenes_infos_temporal_val.pkl}"
TRAIN_SAMPLES_PER_GPU="${TRAIN_SAMPLES_PER_GPU:-1}"
TRAIN_WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU:-2}"
VAL_SAMPLES_PER_GPU="${VAL_SAMPLES_PER_GPU:-1}"
VAL_WORKERS_PER_GPU="${VAL_WORKERS_PER_GPU:-2}"
OFFLINE_UNIQUE_ANCHOR="${OFFLINE_UNIQUE_ANCHOR:-false}"
SERIAL_LR_POLICY="${SERIAL_LR_POLICY:-config}"
SERIAL_FIXED_LR="${SERIAL_FIXED_LR:-}"
VAL_LOG_SUBDIR="${VAL_LOG_SUBDIR:-val_logs}"
VAL_LOAD_REPORT_SUBDIR="${VAL_LOAD_REPORT_SUBDIR:-align_val_load_reports}"
VAL_SUMMARY_FILE="${VAL_SUMMARY_FILE:-val_metrics.tsv}"
VAL_METRICS_SUBDIR="${VAL_METRICS_SUBDIR:-val_metrics_json}"

# Auto export/plot compare artifacts after each epoch validation.
AUTO_EXPORT_COMPARE="${AUTO_EXPORT_COMPARE:-true}"
COMPARE_EXPORT_SCRIPT="${COMPARE_EXPORT_SCRIPT:-tools/export_train_val_compare.py}"
COMPARE_PLOT_SCRIPT="${COMPARE_PLOT_SCRIPT:-tools/plot_train_val_compare.py}"
COMPARE_MD_SCRIPT="${COMPARE_MD_SCRIPT:-tools/tsv_to_markdown.py}"
COMPARE_TSV_NAME="${COMPARE_TSV_NAME:-train_val_compare.tsv}"
COMPARE_PNG_NAME="${COMPARE_PNG_NAME:-train_val_compare.png}"
COMPARE_MD_NAME="${COMPARE_MD_NAME:-train_val_compare.md}"
COMPARE_MD_COLUMNS="${COMPARE_MD_COLUMNS:-epoch,train_last_loss_align,train_avg_loss_align,val_loss_align,train_last_acc_i2t_top1,val_i2t_top1,train_last_acc_t2i_top1,val_t2i_top1}"
COMPARE_TRAIN_LOG="${COMPARE_TRAIN_LOG:-}"

# Run validation on one GPU by default (usually less contention and simpler).
VAL_CUDA_VISIBLE_DEVICES="${VAL_CUDA_VISIBLE_DEVICES:-$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)}"

# Epoch-range consistency marker used by multi-node startup guard.
EPOCH_RANGE_MARKER="${WORK_DIR}/.ddp_epoch_range.env"

mkdir -p "${WORK_DIR}"
mkdir -p "${WORK_DIR}/${VAL_LOG_SUBDIR}"
mkdir -p "${WORK_DIR}/${VAL_LOAD_REPORT_SUBDIR}"
mkdir -p "${WORK_DIR}/${VAL_METRICS_SUBDIR}"

summary_path="${WORK_DIR}/${VAL_SUMMARY_FILE}"
if [[ ! -f "${summary_path}" ]]; then
  printf "epoch\talign_ckpt\tval_loss_align\ti2t_top1\tt2i_top1\ti2t_r5\ti2t_r10\tt2i_r5\tt2i_r10\tval_log\tload_report\n" > "${summary_path}"
fi

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Dataset  : ${DATASET_PROFILE}"
echo "[INFO] Work dir : ${WORK_DIR}"
echo "[INFO] Epochs   : ${START_EPOCH} -> ${END_EPOCH}"
echo "[INFO] Train horizon: ${SERIAL_TOTAL_EPOCHS}"
echo "[INFO] DDP GPUs : ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "[INFO] DDP nodes: nnodes=${NNODES}, node_rank=${NODE_RANK}"
if [[ "${TORCHRUN_BACKEND_MODE}" == "static" ]]; then
  echo "[INFO] Torchrun mode: static (master=${MASTER_ADDR}:${MASTER_PORT})"
else
  echo "[INFO] Torchrun mode: c10d (rdzv=${MASTER_ADDR}:${MASTER_PORT}, rdzv_id=${RDZV_ID}, timeout=${TORCHRUN_RDZV_TIMEOUT}s)"
fi
echo "[INFO] Rsync sync: enable=${ENABLE_RSYNC_SYNC}, target=${RSYNC_TARGET_USER}@${RSYNC_TARGET_IP}::${RSYNC_TARGET_PATH}"
echo "[INFO] Scene json: ${SCENE_JSON}"
echo "[INFO] Train ann : ${TRAIN_ANN}"
echo "[INFO] Val ann   : ${VAL_ANN}"
echo "[INFO] Unique anchor: ${OFFLINE_UNIQUE_ANCHOR}"
echo "[INFO] Serial LR policy: ${SERIAL_LR_POLICY}"
if [[ -n "${SERIAL_FIXED_LR}" ]]; then
  echo "[INFO] Serial fixed LR override: ${SERIAL_FIXED_LR}"
fi
echo "[INFO] Val logs  : ${WORK_DIR}/${VAL_LOG_SUBDIR}"
echo "[INFO] Val summary: ${summary_path}"
echo "[INFO] Val metrics json: ${WORK_DIR}/${VAL_METRICS_SUBDIR}"
echo "[INFO] Auto compare export: ${AUTO_EXPORT_COMPARE}"
if [[ "${NNODES}" -gt 1 ]]; then
  echo "[INFO] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<empty>}"
  echo "[INFO] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<empty>}"
fi

if [[ "${START_EPOCH}" -gt "${END_EPOCH}" ]]; then
  echo "[ERROR] START_EPOCH (${START_EPOCH}) must be <= END_EPOCH (${END_EPOCH})"
  exit 1
fi

if [[ "${SERIAL_TOTAL_EPOCHS}" -lt "${END_EPOCH}" ]]; then
  echo "[ERROR] SERIAL_TOTAL_EPOCHS (${SERIAL_TOTAL_EPOCHS}) must be >= END_EPOCH (${END_EPOCH})"
  exit 1
fi

if [[ "${SERIAL_LR_POLICY}" != "config" && "${SERIAL_LR_POLICY}" != "Fixed" ]]; then
  echo "[ERROR] SERIAL_LR_POLICY must be config or Fixed, got: ${SERIAL_LR_POLICY}"
  exit 1
fi

if [[ "${NNODES}" -gt 1 && "${REQUIRE_SOCKET_IFNAME}" == "true" ]]; then
  if [[ -z "${GLOO_SOCKET_IFNAME}" || -z "${NCCL_SOCKET_IFNAME}" ]]; then
    echo "[ERROR] Multi-node run requires explicit NIC binding."
    echo "[ERROR] Please set both GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME (e.g. eno2)."
    exit 1
  fi
fi

wait_for_resume_ckpt() {
  local ckpt="$1"
  local waited=0
  while [[ ! -f "${ckpt}" ]]; do
    if [[ "${waited}" -ge "${RESUME_WAIT_SECONDS}" ]]; then
      echo "[ERROR] Timed out waiting for resume checkpoint: ${ckpt}"
      echo "[ERROR] waited=${waited}s, RESUME_WAIT_SECONDS=${RESUME_WAIT_SECONDS}"
      return 1
    fi
    if (( waited % 60 == 0 )); then
      echo "[INFO] Waiting for checkpoint sync: ${ckpt} (waited ${waited}s)"
    fi
    sleep "${RESUME_WAIT_INTERVAL}"
    waited=$((waited + RESUME_WAIT_INTERVAL))
  done
  return 0
}

sync_required_ckpt_to_peer() {
  local ckpt="$1"
  if [[ "${NNODES}" -le 1 ]]; then
    return 0
  fi
  if [[ "${NODE_RANK}" -ne 0 ]]; then
    return 0
  fi
  if [[ "${ENABLE_RSYNC_SYNC}" != "true" ]]; then
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[ERROR] ENABLE_RSYNC_SYNC=true but rsync command not found."
    return 1
  fi
  if [[ ! -f "${RSYNC_PASSWORD_FILE}" ]]; then
    echo "[ERROR] ENABLE_RSYNC_SYNC=true but RSYNC_PASSWORD_FILE not found: ${RSYNC_PASSWORD_FILE}"
    return 1
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Required checkpoint not found for sync: ${ckpt}"
    return 1
  fi

  echo "[INFO] Sync required checkpoint to peer: ${ckpt}"
  rsync ${RSYNC_EXTRA_OPTS} --password-file="${RSYNC_PASSWORD_FILE}" \
    "${ckpt}" \
    "${RSYNC_TARGET_USER}@${RSYNC_TARGET_IP}::${RSYNC_TARGET_PATH}"
}

sync_epoch_range_marker_to_peer() {
  local marker="$1"
  if [[ "${NNODES}" -le 1 ]]; then
    return 0
  fi
  if [[ "${NODE_RANK}" -ne 0 ]]; then
    return 0
  fi
  if [[ "${ENABLE_RSYNC_SYNC}" != "true" ]]; then
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[ERROR] ENABLE_RSYNC_SYNC=true but rsync command not found."
    return 1
  fi
  if [[ ! -f "${RSYNC_PASSWORD_FILE}" ]]; then
    echo "[ERROR] ENABLE_RSYNC_SYNC=true but RSYNC_PASSWORD_FILE not found: ${RSYNC_PASSWORD_FILE}"
    return 1
  fi
  if [[ ! -f "${marker}" ]]; then
    echo "[ERROR] Epoch-range marker not found for sync: ${marker}"
    return 1
  fi

  echo "[INFO] Sync epoch-range marker to peer: ${marker}"
  rsync ${RSYNC_EXTRA_OPTS} --password-file="${RSYNC_PASSWORD_FILE}" \
    "${marker}" \
    "${RSYNC_TARGET_USER}@${RSYNC_TARGET_IP}::${RSYNC_TARGET_PATH}"
}

wait_for_epoch_range_marker() {
  local marker="$1"
  local waited=0
  while [[ ! -f "${marker}" ]]; do
    if [[ "${waited}" -ge "${RESUME_WAIT_SECONDS}" ]]; then
      echo "[ERROR] Timed out waiting for epoch-range marker: ${marker}"
      echo "[ERROR] waited=${waited}s, RESUME_WAIT_SECONDS=${RESUME_WAIT_SECONDS}"
      return 1
    fi
    if (( waited % 60 == 0 )); then
      echo "[INFO] Waiting for epoch-range marker: ${marker} (waited ${waited}s)"
    fi
    sleep "${RESUME_WAIT_INTERVAL}"
    waited=$((waited + RESUME_WAIT_INTERVAL))
  done
  return 0
}

prepare_and_validate_epoch_range_marker() {
  local marker="$1"

  if [[ "${NNODES}" -le 1 ]]; then
    return 0
  fi

  if [[ "${NODE_RANK}" -eq 0 ]]; then
    cat > "${marker}" <<EOF
START_EPOCH=${START_EPOCH}
END_EPOCH=${END_EPOCH}
SERIAL_TOTAL_EPOCHS=${SERIAL_TOTAL_EPOCHS}
DATASET_PROFILE=${DATASET_PROFILE}
RDZV_ID=${RDZV_ID}
EOF
    sync_epoch_range_marker_to_peer "${marker}"
    return 0
  fi

  if [[ "${ENABLE_RSYNC_SYNC}" == "true" ]]; then
    wait_for_epoch_range_marker "${marker}"
  fi

  if [[ ! -f "${marker}" ]]; then
    echo "[WARN] Epoch-range marker is unavailable on node_rank=${NODE_RANK}: ${marker}"
    echo "[WARN] Skip strict epoch-range cross-node validation for this run."
    return 0
  fi

  marker_start="$(grep -E '^START_EPOCH=' "${marker}" | head -1 | cut -d'=' -f2-)"
  marker_end="$(grep -E '^END_EPOCH=' "${marker}" | head -1 | cut -d'=' -f2-)"
  marker_total="$(grep -E '^SERIAL_TOTAL_EPOCHS=' "${marker}" | head -1 | cut -d'=' -f2-)"
  marker_profile="$(grep -E '^DATASET_PROFILE=' "${marker}" | head -1 | cut -d'=' -f2-)"
  marker_rdzv_id="$(grep -E '^RDZV_ID=' "${marker}" | head -1 | cut -d'=' -f2-)"

  if [[ "${marker_start}" != "${START_EPOCH}" || "${marker_end}" != "${END_EPOCH}" || "${marker_total}" != "${SERIAL_TOTAL_EPOCHS}" || "${marker_profile}" != "${DATASET_PROFILE}" || "${marker_rdzv_id}" != "${RDZV_ID}" ]]; then
    echo "[ERROR] Cross-node startup mismatch detected on node_rank=${NODE_RANK}."
    echo "[ERROR] Local : START_EPOCH=${START_EPOCH}, END_EPOCH=${END_EPOCH}, SERIAL_TOTAL_EPOCHS=${SERIAL_TOTAL_EPOCHS}, DATASET_PROFILE=${DATASET_PROFILE}, RDZV_ID=${RDZV_ID}"
    echo "[ERROR] Marker: START_EPOCH=${marker_start}, END_EPOCH=${marker_end}, SERIAL_TOTAL_EPOCHS=${marker_total}, DATASET_PROFILE=${marker_profile}, RDZV_ID=${marker_rdzv_id}"
    exit 1
  fi
}

prepare_and_validate_epoch_range_marker "${EPOCH_RANGE_MARKER}"

for epoch in $(seq "${START_EPOCH}" "${END_EPOCH}"); do
  prev_epoch=$((epoch - 1))

  echo ""
  echo "[INFO] ===== Epoch ${epoch}: train ====="
  echo "[INFO] Epoch ${epoch} mode: offline_unique_anchor=${OFFLINE_UNIQUE_ANCHOR}"

  train_cmd=(
    torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    tools/train.py
    "${CONFIG}"
    --launcher pytorch
    --work-dir "${WORK_DIR}"
    --no-validate
  )

  if [[ "${TORCHRUN_BACKEND_MODE}" == "static" ]]; then
    train_cmd=(
      torchrun
      --nnodes="${NNODES}"
      --node_rank="${NODE_RANK}"
      --nproc_per_node="${NPROC_PER_NODE}"
      --rdzv_backend=static
      --master_addr="${MASTER_ADDR}"
      --master_port="${MASTER_PORT}"
      tools/train.py
      "${CONFIG}"
      --launcher pytorch
      --work-dir "${WORK_DIR}"
      --no-validate
    )
  else
    train_cmd=(
      torchrun
      --nnodes="${NNODES}"
      --node_rank="${NODE_RANK}"
      --nproc_per_node="${NPROC_PER_NODE}"
      --rdzv_backend=c10d
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      --rdzv_id="${RDZV_ID}"
      --rdzv_conf "timeout=${TORCHRUN_RDZV_TIMEOUT},is_host=$([[ "${NODE_RANK}" -eq 0 ]] && echo true || echo false)${TORCHRUN_RDZV_CONF_EXTRA:+,${TORCHRUN_RDZV_CONF_EXTRA}}"
      tools/train.py
      "${CONFIG}"
      --launcher pytorch
      --work-dir "${WORK_DIR}"
      --no-validate
    )
  fi

  if [[ "${epoch}" -gt 1 ]]; then
    resume_ckpt="${WORK_DIR}/epoch_${prev_epoch}.pth"
    if [[ "${NNODES}" -gt 1 && "${NODE_RANK}" -ne 0 && "${ENABLE_RSYNC_SYNC}" == "true" ]]; then
      wait_for_resume_ckpt "${resume_ckpt}"
    fi
    if [[ ! -f "${resume_ckpt}" ]]; then
      echo "[ERROR] Resume checkpoint not found: ${resume_ckpt}"
      exit 1
    fi
    train_cmd+=(--resume-from "${resume_ckpt}")
  fi

  train_cmd+=(
    --cfg-options
    model.run_mode=offline_train
    model.offline_split=train
    model.scene_json="${SCENE_JSON}"
    data.train.ann_file="${TRAIN_ANN}"
    data.train.mono_cfg=None
    data.train.offline_meta_only=True
    data.train.offline_unique_anchor="${OFFLINE_UNIQUE_ANCHOR}"
    data.samples_per_gpu="${TRAIN_SAMPLES_PER_GPU}"
    data.workers_per_gpu="${TRAIN_WORKERS_PER_GPU}"
    data.persistent_workers=False
    data.prefetch_factor=1
    total_epochs="${epoch}"
    runner.max_epochs="${epoch}"
  )

  if [[ "${SERIAL_LR_POLICY}" == "Fixed" ]]; then
    train_cmd+=(lr_config.policy=Fixed)
    if [[ -n "${SERIAL_FIXED_LR}" ]]; then
      train_cmd+=(optimizer.lr="${SERIAL_FIXED_LR}")
    fi
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${train_cmd[@]}"

  # In multi-node runs, only rank0 performs validation/export/write-back.
  # Other nodes return to the next epoch training command and wait at rendezvous.
  if [[ "${NNODES}" -gt 1 && "${NODE_RANK}" -ne 0 ]]; then
    echo "[INFO] node_rank=${NODE_RANK}: skip validate/export for epoch ${epoch}; wait for next train rendezvous."
    continue
  fi

  align_ckpt="${WORK_DIR}/align_trainable_epoch_${epoch}.pth"
  if [[ ! -f "${align_ckpt}" ]]; then
    echo "[ERROR] Align checkpoint not found after training: ${align_ckpt}"
    echo "[INFO] Existing align checkpoints:"
    ls -1 "${WORK_DIR}"/align_trainable_epoch_*.pth 2>/dev/null || true
    exit 1
  fi

  # Keep non-shared multi-node WORK_DIR in sync for next epoch resume.
  sync_required_ckpt_to_peer "${WORK_DIR}/epoch_${epoch}.pth"

  echo "[INFO] ===== Epoch ${epoch}: validate ====="
  echo "[INFO] Epoch ${epoch} mode: offline_unique_anchor=${OFFLINE_UNIQUE_ANCHOR}"
  val_log="${WORK_DIR}/${VAL_LOG_SUBDIR}/epoch_${epoch}.log"
  load_report="${WORK_DIR}/${VAL_LOAD_REPORT_SUBDIR}/align_trainable_epoch_${epoch}.json"
  metrics_json="${WORK_DIR}/${VAL_METRICS_SUBDIR}/epoch_${epoch}.json"
  val_cfg_opts=(
    model.run_mode=offline_infer_validate
    model.offline_split=val
    model.scene_json="${SCENE_JSON}"
    data.val.offline_meta_only=True
    data.val.offline_unique_anchor="${OFFLINE_UNIQUE_ANCHOR}"
  )
  if [[ -n "${VAL_ANN}" ]]; then
    val_cfg_opts+=(data.val.ann_file="${VAL_ANN}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${VAL_CUDA_VISIBLE_DEVICES}" \
  python tools/validate_vlm_align.py \
    "${CONFIG}" \
    --base-ckpt "${BASE_CKPT}" \
    --align-ckpt "${align_ckpt}" \
    --load-report "${load_report}" \
    --samples-per-gpu "${VAL_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${VAL_WORKERS_PER_GPU}" \
    --cfg-options \
    "${val_cfg_opts[@]}" \
    2>&1 | tee "${val_log}"
  val_rc=${PIPESTATUS[0]}
  set -e

  val_loss_align=$(grep -E '^val_loss_align:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  i2t_top1=$(grep -E '^i2t_top1:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  t2i_top1=$(grep -E '^t2i_top1:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  i2t_r5=$(grep -E '^i2t_r5:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  i2t_r10=$(grep -E '^i2t_r10:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  t2i_r5=$(grep -E '^t2i_r5:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')
  t2i_r10=$(grep -E '^t2i_r10:' "${val_log}" | tail -1 | awk -F': ' '{print $2}')

  val_loss_align=${val_loss_align:-NA}
  i2t_top1=${i2t_top1:-NA}
  t2i_top1=${t2i_top1:-NA}
  i2t_r5=${i2t_r5:-NA}
  i2t_r10=${i2t_r10:-NA}
  t2i_r5=${t2i_r5:-NA}
  t2i_r10=${t2i_r10:-NA}

  cat > "${metrics_json}" <<EOF
{
  "epoch": ${epoch},
  "align_ckpt": "${align_ckpt}",
  "load_report": "${load_report}",
  "val_log": "${val_log}",
  "val_return_code": ${val_rc},
  "val_loss_align": "${val_loss_align}",
  "i2t_top1": "${i2t_top1}",
  "t2i_top1": "${t2i_top1}",
  "i2t_r5": "${i2t_r5}",
  "i2t_r10": "${i2t_r10}",
  "t2i_r5": "${t2i_r5}",
  "t2i_r10": "${t2i_r10}"
}
EOF

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${epoch}" "${align_ckpt}" "${val_loss_align}" "${i2t_top1}" "${t2i_top1}" \
    "${i2t_r5}" "${i2t_r10}" "${t2i_r5}" "${t2i_r10}" "${val_log}" "${load_report}" \
    >> "${summary_path}"

  echo "[INFO] Saved val log: ${val_log}"
  echo "[INFO] Updated val summary: ${summary_path}"
  echo "[INFO] Saved val metrics json: ${metrics_json}"

  if [[ "${AUTO_EXPORT_COMPARE}" == "true" ]]; then
    if [[ "${COMPARE_TSV_NAME}" = /* ]]; then
      compare_tsv_output_arg="${COMPARE_TSV_NAME}"
      compare_tsv_path="${COMPARE_TSV_NAME}"
    else
      compare_tsv_output_arg="${COMPARE_TSV_NAME}"
      compare_tsv_path="${WORK_DIR}/${COMPARE_TSV_NAME}"
    fi

    if [[ "${COMPARE_PNG_NAME}" = /* ]]; then
      compare_png_path="${COMPARE_PNG_NAME}"
    else
      compare_png_path="${WORK_DIR}/${COMPARE_PNG_NAME}"
    fi

    if [[ "${COMPARE_MD_NAME}" = /* ]]; then
      compare_md_path="${COMPARE_MD_NAME}"
    else
      compare_md_path="${WORK_DIR}/${COMPARE_MD_NAME}"
    fi

    export_cmd=(
      python "${COMPARE_EXPORT_SCRIPT}"
      --work-dir "${WORK_DIR}"
      --output "${compare_tsv_output_arg}"
    )
    if [[ -n "${COMPARE_TRAIN_LOG}" ]]; then
      export_cmd+=(--train-log "${COMPARE_TRAIN_LOG}")
    fi

    set +e
    "${export_cmd[@]}"
    export_rc=$?
    if [[ "${export_rc}" -eq 0 ]]; then
      python "${COMPARE_PLOT_SCRIPT}" \
        --compare-file "${compare_tsv_path}" \
        --output "${compare_png_path}"
      plot_rc=$?

      python "${COMPARE_MD_SCRIPT}" \
        --input "${compare_tsv_path}" \
        --output "${compare_md_path}" \
        --columns "${COMPARE_MD_COLUMNS}"
      md_rc=$?
    else
      plot_rc=1
      md_rc=1
    fi
    set -e

    if [[ "${export_rc}" -eq 0 && "${plot_rc}" -eq 0 && "${md_rc}" -eq 0 ]]; then
      echo "[INFO] Refreshed compare table: ${compare_tsv_path}"
      echo "[INFO] Refreshed compare plot : ${compare_png_path}"
      echo "[INFO] Refreshed compare md   : ${compare_md_path}"
    else
      echo "[WARN] Auto compare refresh failed (export_rc=${export_rc}, plot_rc=${plot_rc}, md_rc=${md_rc})."
    fi
  fi

  if [[ "${val_rc}" -ne 0 ]]; then
    echo "[ERROR] Validation failed at epoch ${epoch} with exit code ${val_rc}."
    echo "[ERROR] Metrics were still saved. Please inspect ${val_log} and ${metrics_json}."
    exit "${val_rc}"
  fi

done

echo ""
if [[ "${NNODES}" -gt 1 && "${NODE_RANK}" -ne 0 ]]; then
  echo "[INFO] Done. Worker node_rank=${NODE_RANK} finished training loop for epochs ${START_EPOCH}..${END_EPOCH}."
else
  echo "[INFO] Done. Serial train+val finished for epochs ${START_EPOCH}..${END_EPOCH}."
fi
