# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class AttentionLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(AttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        visual_predict = predicts['visual_out']
        predict = predicts['predict']
        targets = batch[1].astype("int64")
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[1], predict.shape[2]

        visual = paddle.reshape(visual_predict, [-1, num_classes])
        fuse = paddle.reshape(predict, [-1, num_classes])
        targets = paddle.reshape(targets, [-1])

        visual_loss = paddle.sum(self.loss_func(visual, targets)) / batch_size
        fuse_loss = paddle.sum(self.loss_func(fuse, targets)) / batch_size

        total_loss = 3 * visual_loss + fuse_loss
        
        return {'loss':total_loss, 'visual_loss':visual_loss, 'fuse_loss':fuse_loss}
