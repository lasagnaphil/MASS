for num_workers in 8 16 32 47
do
    for frag_len in 64 128 256 512
    do
        timeout 3h python python/ray_train.py --config=ppo-${num_workers}-${frag_len}
    done
done
