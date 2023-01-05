# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle
import paddle.distributed as dist

paddle.seed(2)

from network.data.dataset import build_dataloader
from network.optimizer.optimizer import build_optimizer
from network.utils.save_load import load_dygraph_params
from network.modeling.model import BaseModel
from network.losses.att_loss import AttentionLoss
from network.postprocess.postprocess import SRNLabelDecode
from network.metrics.metric import RecMetric

import tools.program as program

dist.get_world_size()


def main(config, device, logger):

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error("No Images in train dataset!\n" )
        return

    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    
    # build post process
    post_process_class = SRNLabelDecode(**config['PostProcess'], mode = 'Train')

    # build model
    char_num = len(getattr(post_process_class, 'character'))
    config['Architecture']["Head"]['out_channels'] = char_num
    model = BaseModel(config['Architecture'])

    # build loss
    loss_class = AttentionLoss(**config['Loss'])

    # build optim
    lr = config['Optimizer']['lr']['learning_rate']
    optimizer = build_optimizer(
        config['Optimizer'],
        lr=lr,
        parameters=model.parameters())

    # build metric
    eval_class = RecMetric(**config['Metric'])
    
    # load pretrain model
    pre_best_model_dict = load_dygraph_params(config, model, logger, optimizer)
    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    logger.info('valid dataloader has {} iters'.format(len(valid_dataloader)))

    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr, post_process_class,
                  eval_class, pre_best_model_dict, logger, len(train_dataloader))


if __name__ == '__main__':
    config, device, logger = program.preprocess(is_train=True)
    main(config, device, logger)
