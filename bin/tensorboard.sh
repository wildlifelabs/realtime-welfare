#!/bin/bash
# (C)2022 J.Cincotta
# Version 3.0
# Written by Joe Cincotta
#
# 
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
LOGROOT="$1"
shift
echo "$LOGROOT"
cd $LOGROOT
trap "exit" INT TERM
trap "kill 0" EXIT
tensorboard --logdir $LOGROOT --host localhost --port 8008 --load_fast false

