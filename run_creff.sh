#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate exFL
cd /media/cgy/disk1/gtafl-github
for S in 1 2 3; do
  echo "########## CReFF seed=$S 开始 $(date +%H:%M:%S) ##########"
  python main.py --test_distribution backward50 --device cuda:0 --seed $S     --num_rounds 200 --num_clients 20 --num_online_clients 8 --mutil_test 2>&1 | tee logs/creff_seed$S.log
  echo "SEED${S}_EXIT=${PIPESTATUS[0]}" >> logs/creff_multi.status
done
echo ALL_DONE >> logs/creff_multi.status
