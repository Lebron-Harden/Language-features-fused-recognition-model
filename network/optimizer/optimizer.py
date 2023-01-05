# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import unicode_literals
import copy
import paddle
from paddle import optimizer as optim

__all__ = ['build_optimizer']


class L2Decay(object):

    def __init__(self, factor=0.0, **kwargs):
        super(L2Decay, self).__init__()
        self.regularization_coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L2Decay(self.regularization_coeff)
        return reg


class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode

    def __call__(self, parameters):
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            parameters=parameters)
        return opt


def build_optimizer(config, lr, parameters):
    config = copy.deepcopy(config)

    reg_config = config.pop('regularizer')

    reg = L2Decay(**reg_config)
    reg = reg()

    
    optim = Adam(learning_rate=lr,
                 weight_decay=reg,
                 grad_clip=None,
                 **config)

    return optim(parameters)
