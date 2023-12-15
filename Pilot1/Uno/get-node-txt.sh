#!/bin/bash
set -eu

# GET NODE TXT
# Pull out one node from a plan txt file

if (( ${#} != 2 ))
then
  echo "get-node-txt: Provide PLAN_JSON NODE"
  exit 1
fi

PLAN_JSON=$1
NODE=$2

THIS=$( dirname $0 )

awk -v node=$NODE -f $THIS/get-node-txt.awk < $PLAN_JSON
