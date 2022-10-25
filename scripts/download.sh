#!/bin/bash

# Downloads the speech dataset into the datasets directory
# USE: ./scripts/download.sh

set -eux

# Uncomment to nuke the existing directory
# rm -rf datasets

if [ ! -e datasets ]; then
  mkdir datasets

  if [ ! -e /tmp/speechdata.zip ]; then
    wget https://osf.io/download/g6q5m/ -O /tmp/speechdata.zip
  fi

  unzip /tmp/speechdata.zip -d datasets

  # Uncomment to cleanup tmp
  # rm /tmp/speechdata.zip
fi
