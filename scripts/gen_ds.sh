cd ..
python nsdf/make_dataset.py \
       --sdf-module example_sdf --sdf-fn sdf \
       --bbox -1. 1. -1. 1. -1 1. \
       --n-uniform 800000 \
       --n-surface 200000 \
       --surface-eps 1 \
       --out data_sphere.npz

