#!/bin/bash

for f in *.json; do pattern=$(echo sed \'s/"seed": [0-9]\+,/"seed": $RANDOM,/g\'); $pattern $f; done
