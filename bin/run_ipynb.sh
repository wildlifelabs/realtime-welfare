#!/bin/bash
# (C)2022 J.Cincotta
# Version 3.0
# Written by Joe Cincotta
#
# usage: run_ipynb.sh <environemtn name> <notebook ipynb filename>
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
NBROOT="$1"
shift
NOTEBOOK="$1"
shift
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "$NBROOT"
cd $NBROOT
for file in *.ipynb
  do
    jupyter trust $file
  done

rm -f pm-log.txt
touch pm-log.txt
tail -f pm-log.txt &
tensorboard --logdir=$NBROOT/checkpoints --host localhost --port=8008 --DEBUG &
papermill $NOTEBOOK $NOTEBOOK -k python3 --log-output --log-level DEBUG --request-save-on-cell-execute --report-mode  --stdout-file pm-log.txt


