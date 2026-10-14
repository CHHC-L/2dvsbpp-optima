#!/bin/sh
# Fetch and build the SAT tools used by src/opp_sat.py / src/opp_certify.py (DRAT certificates).
# Tested with CaDiCaL 3.0.1 and drat-trim (GitHub master of 2026-10-14).
# usage (from the project root): sh scripts/get_sat_tools.sh
set -e
mkdir -p "$(dirname "$0")/../third_party"
cd "$(dirname "$0")/../third_party"
[ -d cadical ] || git clone --depth 1 --branch rel-3.0.1 https://github.com/arminbiere/cadical
(cd cadical && [ -x build/cadical ] || (./configure && make -j4))
[ -d drat-trim ] || git clone --depth 1 https://github.com/marijnheule/drat-trim
(cd drat-trim && [ -x drat-trim ] || cc -O2 -o drat-trim drat-trim.c)
echo "cadical: $(./cadical/build/cadical --version)"; ls -l drat-trim/drat-trim
