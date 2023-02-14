# python3 supervised.py --dataset_name twonormal  --path tmp3 --density.TYPE RND --env_cfg.embed_dim 8 --density.normalizer ema --vis.scale 10. 
# python3 supervised.py --dataset_name twonormal  --path tmp2/rnd0 --density.TYPE RND --env_cfg.embed_dim 0 --density.normalizer ema --max_epoch 20 --vis.scale 10.
# python3 supervised.py --dataset_name twonormal  --path tmp2/rnd8 --density.TYPE RND --env_cfg.embed_dim 8 --density.normalizer ema --max_epoch 20 --vis.scale 10.
python3 supervised.py --dataset_name twonormal  --path tmp2/vae0 --density.TYPE VAE --env_cfg.embed_dim 0 --density.normalizer none --max_epoch 20 --vis.scale 1.
python3 supervised.py --dataset_name twonormal  --path tmp2/vae8 --density.TYPE VAE --env_cfg.embed_dim 8 --density.normalizer none --max_epoch 20
#python3 supervised.py --dataset_name twonormal