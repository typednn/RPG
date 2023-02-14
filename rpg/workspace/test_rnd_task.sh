# test performance of rnd on different envs  
#python3 tester/test_new_triple.py --var z2 --rnd.scale 1. --reward_scale 0. --path tmp/z2rnd
#python3 tester/test_new_triple.py --var normal --rnd.scale 1. --reward_scale 0. --path tmp/normalrnd --epoch 1000
#python3 tester/test_new_triple.py --var normal --rnd.scale 1. --reward_scale 0. --path tmp/normalrnd2 --epoch 1000 --info.coef 1.

#TODO: make the entropy/std larger

python3 tester/test_new_triple.py --var normal --rnd.scale 1. --reward_scale 0. --path tmp/normalrnd --epoch 1000 --head.squash True --pi_a.ent.coef 0.1