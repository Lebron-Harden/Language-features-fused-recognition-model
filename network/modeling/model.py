# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from network.modeling.resnet import ResNet
from network.modeling.fuse_head import FuseHead


__all__ = ['BaseModel']


class BaseModel(nn.Layer):
    def __init__(self, config):

        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        
        config["Backbone"]['in_channels'] = in_channels

        self.backbone = ResNet(**config["Backbone"])

        in_channels = self.backbone.out_channels

        config["Head"]['in_channels'] = in_channels

        self.head = FuseHead(**config["Head"])


    def forward(self, x, data=None):

        x = self.backbone(x)

        x = self.head(x, targets=data)
        
        return x
