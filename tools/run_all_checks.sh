#!/usr/bin/env bash
# Every offline check of the harness in one command. No LLM is called, no tokens are spent (the LLM variables are
# removed from the environment of each check).
#
#   tools/run_all_checks.sh [SAMPLES_DIR] [--quick] [--docker]
#
#   SAMPLES_DIR  the organisers' sample task directories (default: $SAMPLES or ../task-dir/task-dir)
#   --quick      skip the slow suites (deterministic-solver sweep, bug zoo): about 40 s instead of about 4 minutes
#   --docker     also build the image and run tools/smoke_image.sh (needs access to Docker)
#
# Exit status 0 only if every suite passed. Needs only the Python 3 standard library.
set -uo pipefail
cd "$(dirname "$0")/.."
SAMPLES="${SAMPLES:-../task-dir/task-dir}"
QUICK=0
DOCKER=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --docker) DOCKER=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) SAMPLES="$a" ;;
  esac
done
if [ ! -d "$SAMPLES" ]; then
  echo "sample tasks directory not found: $SAMPLES (pass it as an argument or set SAMPLES)" >&2
  exit 2
fi
SAMPLES="$(realpath "$SAMPLES")"
for t in math search data_analysis coding; do  # a suite that finds no samples of its type would pass without checking them
  if ! ls -d "$SAMPLES"/${t}_* >/dev/null 2>&1; then
    echo "no ${t}_* sample task directories in $SAMPLES" >&2
    exit 2
  fi
done

PASS=0
FAIL=0
SUMMARY=()
run() {  # run "name" command...
  local name="$1"; shift
  local t0 out rc dt
  t0=$(date +%s)
  out=$(env -u LLM_API_KEY -u LLM_BASE_URL "$@" 2>&1)
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then
    PASS=$((PASS + 1)); SUMMARY+=("PASS  ${dt}s  $name")
    printf '  PASS  %4ss  %s\n' "$dt" "$name"
  else
    FAIL=$((FAIL + 1)); SUMMARY+=("FAIL  ${dt}s  $name")
    printf '  FAIL  %4ss  %s\n' "$dt" "$name"
    printf '%s\n' "$out" | tail -15 | sed 's/^/        | /'
  fi
}

echo "samples: $SAMPLES"
echo "== fast suites"
run "edge cases (71 borderline cases of the detectors)"                python3 tools/check_edge_cases.py
run "search: mentor chains, paraphrases (real kb, independent walk)"    python3 -I tools/check_search_chains.py "$SAMPLES"
run "data_analysis: composable filters (real sales.csv)"                python3 -I tools/check_data_filters.py "$SAMPLES"
run "math: random stories vs independent simulation"                    python3 -I tools/check_math_chains.py
run "math: the check fails when the solver is sabotaged (--selftest)"   python3 -I tools/check_math_chains.py --selftest
run "coding: mutation repair (29 fixable, 4 not, real samples)"         python3 -I tools/check_repair.py --real "$SAMPLES"
if [ "$QUICK" -eq 0 ]; then
  echo "== slow suites"
  run "deterministic solvers: 920 solvable / 400 unsolvable + samples"  python3 tools/check_det.py --seeds 20 --extended --samples "$SAMPLES"
  run "coding bug zoo, mutation only (>= 85%)"                          python3 -I tools/check_bug_zoo.py --no-canon --min-rate 0.85
  run "coding bug zoo, full flow (>= 93%)"                              python3 -I tools/check_bug_zoo.py --min-rate 0.93
fi
if [ "$DOCKER" -eq 1 ]; then
  echo "== image"
  run "image: build, size, offline run with the organisers' flags"      tools/smoke_image.sh "$SAMPLES"
fi

echo
echo "== summary"
printf '%s\n' "${SUMMARY[@]}"
echo "passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
