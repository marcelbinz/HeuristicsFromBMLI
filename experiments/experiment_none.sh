#!/bin/bash
python models/train.py --alpha 0 --save-path 'trained_models/none_pretrained_'
python models/train.py --alpha 0.1 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_1_'
python models/train.py --alpha 0.03 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_2_'
python models/train.py --alpha 0.01 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_3_'
python models/train.py --alpha 0.003 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_4_'
python models/train.py --alpha 0.001 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_5_'
python models/train.py --alpha 0.0003 --load-path 'trained_models/none_pretrained_0.pth' --save-path 'trained_models/none_6_'
