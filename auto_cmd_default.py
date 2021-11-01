#-*-coding:utf-8-*-

cmds = [
    "python trainval_net.py --save_dir weights_cag --dataset fake_sim10k --net vgg16  --bs 2 --nw 0 --lr 0.001 --cuda --cag",
    "python trainval_net.py --save_dir weights_cag --dataset real_citysacpes --net vgg16  --bs 2 --nw 0 --lr 0.001 --cuda --cag"
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
