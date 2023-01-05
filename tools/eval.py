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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from network.data.dataset import build_dataloader
from network.utils.save_load import load_dygraph_params
from network.modeling.model import BaseModel
from network.postprocess.postprocess import SRNLabelDecode
from network.metrics.metric import RecMetric

import tools.program as program


def main():

    # build dataloader
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # build post process
    post_process_class = SRNLabelDecode(**config['PostProcess'], mode = 'Eval')

    # build model
    char_num = len(getattr(post_process_class, 'character'))
    config['Architecture']["Head"]['out_channels'] = char_num
    model = BaseModel(config['Architecture'])

    model_type = config['Architecture']['model_type']

    best_model_dict = load_dygraph_params(config, model, logger, None)
    if len(best_model_dict):
        logger.info('metric in ckpt ***************')
        for k, v in best_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # build metric
    eval_class = RecMetric(config['Metric'])

    # start eval
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, model_type)
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    config, device, logger = program.preprocess()
    main()
