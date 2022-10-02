python3 download_from_nautilus.py --dataset simple_envhub
python3 ../dataset/process_traj3.py --name simple_envhub --env_type SimpleEnv
python3 upload2nautilus.py --dataset simple_envhub