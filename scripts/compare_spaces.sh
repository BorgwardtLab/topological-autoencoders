#!/usr/bin/env bash
#
# $1 = data set
# $2 = latent space
# $3 = batch size
#
# Usage: /compare_spaces.sh /tmp/data.csv /tmp/latents.csv 64

DISTANCES=()

for i in `seq 1 10`; do
  shuf -n $3 $1 > /tmp/vr_data_tmp.csv
  vietoris_rips -n /tmp/vr_data_tmp.csv 10000 1 > /tmp/vr_D1.txt 2> /dev/null

  shuf -n $3 $2 > /tmp/vr_latent_tmp.csv
  vietoris_rips -n /tmp/vr_latent_tmp.csv 10000 1 > /tmp/vr_D2.txt 2> /dev/null
  distance=`topological_distance --bottleneck /tmp/vr_D1.txt /tmp/vr_D2.txt 2>/dev/null | head -n 1  | cut -f 2 -d " "`
   DISTANCES+=($distance)
done

# Mean (do we want a standard deviation as well?)
echo ${DISTANCES[@]} | tr ' ' '\n' | awk '{sum+=$1; n++};END{print sum/n}'
