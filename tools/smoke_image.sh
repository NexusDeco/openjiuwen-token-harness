#!/usr/bin/env bash
# Build the image and smoke-test it the way the organisers will run it.
#
#   tools/smoke_image.sh [path-to-the-sample-tasks]        (default: $SAMPLES or ../task-dir/task-dir)
#   DRY_RUN=1 tools/smoke_image.sh ...                     (only print the docker commands)
#
# Needs Docker access (your user in the `docker` group). Steps:
#   1. docker build with timing (the organisers allow 1000 s)
#   2. image size against the 4 GB limit
#   3. offline run on the samples with the organisers' flags (read-only FS, cap-drop ALL, no network). The samples are
#      settled by the deterministic solvers, so no LLM is needed; this also proves openjiuwen imports in the container
#   4. the same run on a copy with ordinary file ownership: root without CAP_DAC_OVERRIDE may fail to write answer.txt
#      there, which is exactly the open question about the permissions of $TASKS
#   5. only if LLM_API_KEY and LLM_BASE_URL are set: two tasks through the real LLM engine inside the container
#      (--no-deterministic, about 1k tokens). Source your env file first:  set -a; source ~/.ojw_env; set +a
set -euo pipefail
cd "$(dirname "$0")/.."
SAMPLES="$(realpath "${1:-${SAMPLES:-../task-dir/task-dir}}")"
IMAGE="${IMAGE:-hack:latest}"
run() { if [ "${DRY_RUN:-}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
FLAGS=(--rm --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp --cap-drop ALL --security-opt no-new-privileges
       --cpus 8 --memory 16g --pids-limit 512 --workdir /app --entrypoint python)
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fresh_copy() {  # $1 = destination; copies the samples (answers are not removed: the harness ignores `expected`)
  rm -rf "$1" && cp -r "$SAMPLES" "$1" && find "$1" \( -name .DS_Store -o -name answer.txt \) -delete
}
answers() { ls "$1"/*/answer.txt 2>/dev/null | wc -l; }

echo "== 1. build"
start=$(date +%s)
run docker build -t "$IMAGE" .
echo "build took $(( $(date +%s) - start )) s (limit 1000 s)"

echo "== 2. size"
if [ "${DRY_RUN:-}" != 1 ]; then
  # The organisers measure exactly this `docker images` line. (`docker image inspect .Size` can differ: it reports only
  # part of the layers with some image stores, so it is not used.)
  line=$(docker images --format "{{.Repository}}:{{.Tag}} -> {{.Size}}" | grep "^${IMAGE} " | head -1 || true)
  echo "$line"
  mb=$(echo "$line" | sed 's/.*-> //' | LC_ALL=C awk '{v=$1; sub(/[A-Za-z]+$/,"",v); u=$1; sub(/^[0-9.]+/,"",u);
        m=(u=="GB")?v*1000:(u=="MB")?v:(u=="kB")?v/1000:(u=="B")?v/1000000:-1; printf "%d", m}')
  if [ -z "${mb:-}" ] || [ "$mb" -lt 0 ]; then verdict="CHECK MANUALLY"; elif [ "$mb" -lt 4000 ]; then verdict=OK; else verdict="TOO BIG"; fi
  echo "size: ${mb:-?} MB (limit 4000 MB): $verdict"
fi

echo "== 3. offline run, files writable for everybody (chmod -R a+rwX)"
T="$WORK/open"; fresh_copy "$T"; chmod -R a+rwX "$T"
run docker run "${FLAGS[@]}" --network none -e LLM_BASE_URL=http://127.0.0.1:9/v1 -e LLM_API_KEY=smoke \
    -v "$T:/data/tasks:rw" "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks
[ "${DRY_RUN:-}" = 1 ] || echo "answer.txt files next to task.json: $(answers "$T") (expected: one per sample task)"

echo "== 4. offline run, files owned by you with default permissions (the permission question)"
T2="$WORK/owned"; fresh_copy "$T2"
run docker run "${FLAGS[@]}" --network none -e LLM_BASE_URL=http://127.0.0.1:9/v1 -e LLM_API_KEY=smoke \
    -v "$T2:/data/tasks:rw" "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks || true
[ "${DRY_RUN:-}" = 1 ] || echo "answer.txt files next to task.json: $(answers "$T2") (0 would mean: root without CAP_DAC_OVERRIDE cannot write there, ask the organisers to make \$TASKS world-writable)"

echo "== 5. real LLM engine on two tasks (optional)"
if [ -n "${LLM_API_KEY:-}" ] && [ -n "${LLM_BASE_URL:-}" ]; then
  T3="$WORK/llm"; mkdir -p "$T3"
  fresh_copy "$WORK/all"; cp -r "$WORK/all/math_0004" "$WORK/all/data_analysis_0002" "$T3/"; chmod -R a+rwX "$T3"
  run docker run "${FLAGS[@]}" -e LLM_BASE_URL -e LLM_API_KEY -e BUDGET=20000 -v "$T3:/data/tasks:rw" "$IMAGE" \
      /app/run_agent.py --tasks-dir /data/tasks --no-deterministic
  [ "${DRY_RUN:-}" = 1 ] || { for t in math_0004 data_analysis_0002; do echo "$t: $(cat "$T3/$t/answer.txt" 2>/dev/null || echo '(no answer)')"; done; echo "(expected: 1735 and 158)"; }
else
  echo "skipped: LLM_API_KEY / LLM_BASE_URL are not set"
fi
