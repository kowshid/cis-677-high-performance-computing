#!/bin/sh
# build.sh -- YOURS to edit. Flags, vectorisation, OpenMP, LTO: all fair game.
#
# It must produce ONE binary at build/task, linking the host harness with your
# solution:
#
#     harness.cpp        host-owned: main, the timing boundary, the checks
#     src/solution.cpp   yours, timed
#
# Your file must not define main(); the harness owns the process.
set -eu

CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:--O3 -std=c++17 -DNDEBUG -fno-math-errno}

f=src/solution.cpp
[ -f "$f" ] || { echo "build.sh: missing $f" >&2; exit 1; }
if grep -Eq '^[[:space:]]*(int|auto)[[:space:]]+main[[:space:]]*\(' "$f"; then
  echo "build.sh: $f defines main(); the harness owns main." >&2
  exit 1
fi

mkdir -p build
# shellcheck disable=SC2086
$CXX $CXXFLAGS -march=native -fopenmp -Iinclude harness.cpp src/solution.cpp -o build/task
echo "built build/task with $CXX $CXXFLAGS -march=native -fopenmp"
