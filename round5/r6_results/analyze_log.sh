#!/bin/bash
# Analyze energy balance from a benchmark log
log="$1"
if [ -z "$log" ]; then
    echo "usage: $0 <logfile>"
    exit 1
fi
echo "=== $log ==="
wc -l "$log"
paste <(grep -E "Step=.*μs" "$log" | sed 's/.*t=\([0-9.]*\) μs.*/\1/') \
      <(grep -E "err=" "$log" | awk -F'err=' '{print $2}' | sed 's/%//') > /tmp/tmp_err.csv
awk 'NR>1 {c++; if($2<5) p++} END{print "All diag points:", c, "| Pass (<5%):", p, "|", int(100*p/c)"%"}' /tmp/tmp_err.csv
awk '$1>=10.0 && $1<=30.0 {c++; if($2<5) p++} END{print "t=10-30μs:", c, "| Pass (<5%):", p, "|", int(100*p/c)"%"}' /tmp/tmp_err.csv
awk '$1>=10.0 {c++; if($2<5) p++; s+=$2} END{print "t>10μs:", c, "| Pass (<5%):", p, "|", int(100*p/c)"%", "| mean err:", s/c"%"}' /tmp/tmp_err.csv
echo "--- Final lines ---"
tail -4 "$log"
