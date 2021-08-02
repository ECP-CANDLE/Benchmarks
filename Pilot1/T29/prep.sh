#!/bin/bash

if [ ! -f "rip.it.test.csv" ]; then
  curl -o rip.it.test.csv.gz ftp://ftp.mcs.anl.gov/pub/candle/public/tutorials/t29res/rip.it.test.csv.gz
  gunzip rip.it.test.csv.gz
fi

if [ ! -f "rip.it.train.csv" ]; then
  curl -o rip.it.train.csv.gz ftp://ftp.mcs.anl.gov/pub/candle/public/tutorials/t29res/rip.it.train.csv.gz
  gunzip rip.it.train.csv.gz
fi