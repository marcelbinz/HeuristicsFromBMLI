#!/bin/bash
python models/train.py --ranking --alpha 0 --save-path 'trained_models/ranking_pretrained_'
python models/train.py --ranking --alpha 0.1 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_1_'
python models/train.py --ranking --alpha 0.03 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_2_'
python models/train.py --ranking --alpha 0.01 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_3_'
python models/train.py --ranking --alpha 0.003 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_4_'
python models/train.py --ranking --alpha 0.001 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_5_'
python models/train.py --ranking --alpha 0.0003 --load-path 'trained_models/ranking_pretrained_0.pth' --save-path 'trained_models/ranking_6_'
