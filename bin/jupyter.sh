#!/bin/bash
# (C)2022 J.Cincotta
# Version 3.0
# Written by Joe Cincotta
#
# 
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
NBROOT="$1"
shift
echo "$NBROOT"

cd $NBROOT
for file in *.ipynb
  do
    jupyter trust $file
  done
echo "9fbaaa1ce20a1159a63c0af8e8dde4695d984616e71169ae"
# Handle killing children on exit
trap "exit" INT TERM
trap "kill 0" EXIT
jupyter lab --no-browser --allow-root --ip 0.0.0.0 --NotebookApp.token='9fbaaa1ce20a1159a63c0af8e8dde4695d984616e71169ae'

