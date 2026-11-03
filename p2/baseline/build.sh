#!/bin/sh
# build.sh -- YOURS to edit. Flags, vectorisation, OpenMP, LTO: all fair game.
#
# It must produce ONE binary at build/task, linking the host harness with your
# two source files:
#
#     harness.cpp        host-owned: main, argv, the timing boundary
#     src/prepare.cpp    yours, untimed
#     src/solution.cpp   yours, timed
#
# Your files must not define main(); the harness owns the process. That is
# checked below, and again by the compiler.
set -eu

CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:--O3 -std=c++17 -DNDEBUG}

for f in src/prepare.cpp src/solution.cpp; do
  [ -f "$f" ] || { echo "build.sh: missing $f" >&2; exit 1; }
  if grep -Eq '^[[:space:]]*(int|auto)[[:space:]]+main[[:space:]]*\(' "$f"; then
    echo "build.sh: $f defines main(); the harness owns main. Your file provides" >&2
    echo "          only hpcbench::prepare or hpcbench::solve." >&2
    exit 1
  fi
done

mkdir -p build

probe() {
  printf 'int main(){return 0;}\n' > build/.probe.cpp
  # shellcheck disable=SC2086
  $CXX $1 build/.probe.cpp -o build/.probe 2>/dev/null && echo "$1" || true
  rm -f build/.probe.cpp build/.probe
}

EXTRA="$(probe -march=native) $(probe -fopenmp)"

# shellcheck disable=SC2086
$CXX $CXXFLAGS $EXTRA -Iinclude \
  harness.cpp src/prepare.cpp src/solution.cpp \
  -o build/task

echo "built build/task with $CXX $CXXFLAGS $EXTRA"
