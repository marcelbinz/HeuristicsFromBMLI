#!/bin/bash
python models/train.py --direction --alpha 0 --save-path 'trained_models/direction_pretrained_'
python models/train.py --direction --alpha 0.1 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_1_'
python models/train.py --direction --alpha 0.03 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_2_'
python models/train.py --direction --alpha 0.01 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_3_'
python models/train.py --direction --alpha 0.003 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_4_'
python models/train.py --direction --alpha 0.001 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_5_'
python models/train.py --direction --alpha 0.0003 --load-path 'trained_models/direction_pretrained_0.pth' --save-path 'trained_models/direction_6_'
