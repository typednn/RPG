# python3 tester/test_new_triple.py --var z2 --save_buffer_epoch 10 --max_epoch 10
# python3 tester/test_new_triple.py --var z2 --fix_buffer tmp/new/buffer.pt --max_epoch 100

# python3 tester/test_new_model.py --save_buffer_epoch 200 --max_epoch 202 --path tmp/cheetah_buffer
# python3 tester/test_new_model.py --fix_buffer tmp/cheetah_buffer/buffer.pt --max_epoch 1000 --path tmp/cheetah_fixedbuffer

python3 maze_tester/test_maze.py --var ant_squash  --reward_scale 0. --path tmp/ant2_buffer --env_name AntMaze2 --max_epoch 8001 --save_buffer_epoch 1000