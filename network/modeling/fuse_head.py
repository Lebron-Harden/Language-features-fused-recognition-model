
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

import paddle
from paddle import nn
from paddle.nn import functional as F

import numpy as np
from .transformer import WrapEncoderForFeature
from .transformer import WrapDecoder

gradient_clip = 10


class Vision(nn.Layer):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, num_decoder_TUs, hidden_dims):
        super(Vision, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_TUs
        self.hidden_dims = hidden_dims

        t = 256
        c = 512
        self.encoder = WrapEncoderForFeature(
            src_vocab_size=1,
            max_length=t,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)
        
        self.visual_linear = paddle.nn.Linear(in_features=in_channels, out_features=self.char_num)

        self.vision_decoder = WrapDecoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)
        
    

    def forward(self, inputs, feature_pos, word_pos, shifted_rd):
        b, c, h, w = inputs.shape
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])

        ## visual encoder
        enc_inputs = [conv_features, feature_pos, None]
        encoder_output = self.encoder(enc_inputs)

        ## visual decoder, link reading order and encoder output

        enc_inputs = [shifted_rd, word_pos, None, encoder_output]
        visual_feature = self.vision_decoder(enc_inputs)
        visual_out = self.visual_linear(visual_feature)
        
        return visual_out, visual_feature


class Language(nn.Layer):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, num_decoder_tus, hidden_dims):
        super(Language, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_tus
        self.hidden_dims = hidden_dims

        self.forward_decoder = WrapDecoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)
        
        self.backward_decoder = WrapDecoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)
        
        self.mul = lambda x: paddle.matmul(x=x,
                                           y=self.forward_decoder.prepare_decoder.emb0.weight,
                                           transpose_y=True)

    def forward(self, word_pos, forward_mask, backward_mask, 
                label, visual_out, rd, shifted_rd):
        
        visual_pred = F.softmax(visual_out, axis=-1)
        visual_pred = paddle.argmax(visual_pred, axis=-1)
        
        forward_rd = shifted_rd
        backward_rd = rd

        '''if self.training:
            import random
            judge = random.randint(0, 10)
            if judge >= 3:
                first_pred = paddle.unsqueeze(visual_pred, axis=-1)
            else:
                first_pred = paddle.unsqueeze(label, axis=-1)
        else:'''
        first_pred = paddle.unsqueeze(visual_pred, axis=-1)


        pad_idx = self.char_num
        shifted_feature = paddle.cast(first_pred, "float32")
        shifted_feature = F.pad(shifted_feature, [1, 0], value=1.0 * pad_idx, data_format="NLC")
        shifted_feature = paddle.cast(shifted_feature, "int64")
        shifted_feature = shifted_feature[:, :-1, :]
        #forward_rd = shifted_feature
        #backward_rd = first_pred


        forward_feature = self.forward_decoder([forward_rd, word_pos, forward_mask, first_pred])
        backward_feature = self.backward_decoder([backward_rd, word_pos, backward_mask, first_pred])
        backward_feature = F.pad(backward_feature, [0, 1], value=0., data_format="NLC")
        backward_feature = backward_feature[:, 1:, ]
        language_feature = forward_feature + backward_feature

        return language_feature, self.mul(language_feature)           


class FuseHead(nn.Layer):
    def __init__(self, in_channels, out_channels, max_text_length, num_heads,
                 num_encoder_TUs, num_decoder_TUs, hidden_dims, **kwargs):
        super(FuseHead, self).__init__()
        self.char_num = out_channels
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_TUs
        self.num_decoder_TUs = num_decoder_TUs
        self.hidden_dims = hidden_dims

        self.vision = Vision(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_TUs=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)

        self.language = Language(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_tus=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)

        self.fuse_linear = paddle.nn.Linear(in_features=self.hidden_dims*2, out_features=self.hidden_dims)

        self.fc = nn.Linear(in_features=self.hidden_dims, out_features=self.char_num)


    def forward(self, inputs, targets=None):
        label = targets[0]
        
        batch_size = inputs.shape[0]
        if inputs.shape[2] != 1:
            feature_dim = inputs.shape[2]*inputs.shape[3]
        else:
            feature_dim = inputs.shape[3]


        ##  prepare input
        forward_mask = np.zeros(shape=(self.max_length, self.max_length), dtype=np.float)
        backward_mask = np.zeros(shape=(self.max_length, self.max_length), dtype=np.float)
        for i in range(forward_mask.shape[0]):
            for j in range(forward_mask.shape[1]):
                if i == j:
                    forward_mask[i][j] = -1000000000
                if i == j:
                    backward_mask[i][j] = -1000000000
        forward_mask = paddle.to_tensor(forward_mask, dtype='float32')
        forward_mask = paddle.unsqueeze(forward_mask, axis=0)
        forward_mask = paddle.unsqueeze(forward_mask, axis=0)
        forward_mask = paddle.tile(forward_mask, [batch_size, self.num_heads, 1, 1])
        backward_mask = paddle.to_tensor(backward_mask, dtype='float32')
        backward_mask = paddle.unsqueeze(backward_mask, axis=0)
        backward_mask = paddle.unsqueeze(backward_mask, axis=0)
        backward_mask = paddle.tile(backward_mask, [batch_size, self.num_heads, 1, 1])
        

        feature_pos = np.array(range(0, feature_dim)).reshape((feature_dim, 1)).astype('int64')
        feature_pos = paddle.to_tensor(feature_pos)
        feature_pos = paddle.unsqueeze(feature_pos, axis=0)
        feature_pos = paddle.tile(feature_pos, (batch_size, 1, 1))


        word_pos = np.array(range(0, self.max_length)).reshape((self.max_length, 1)).astype('int64')
        word_pos = paddle.to_tensor(word_pos)
        word_pos = paddle.unsqueeze(word_pos, axis=0)
        word_pos = paddle.tile(word_pos, (batch_size, 1, 1))


        reading_order = np.array(range(0, self.max_length)).reshape((self.max_length, 1)).astype('int64')
        reading_order = paddle.to_tensor(reading_order)
        reading_order = paddle.unsqueeze(reading_order, axis=0)
        rd = paddle.tile(reading_order, (batch_size, 1, 1))

        shifted_rd = paddle.cast(rd, "float32")
        shifted_rd = F.pad(shifted_rd, [1, 0], value=1.0 * self.char_num, data_format="NLC")
        shifted_rd = paddle.cast(shifted_rd, "int64")
        shifted_rd = shifted_rd[:, :-1, :] 

        others = targets[-4:]
        #forward_mask = others[2]
        #backward_mask = others[3]


        ## vision part
        visual_out, visual_feature = self.vision(inputs, feature_pos, word_pos, shifted_rd)
        
        ## language part
        language_feature, _ = self.language(word_pos, forward_mask, backward_mask, 
                                            label, visual_out, rd=rd, shifted_rd=shifted_rd)

        ## fuse
        combine_feature = paddle.concat([visual_feature, language_feature], axis=-1)
        combine_feature = self.fuse_linear(combine_feature)
        combine_feature = F.sigmoid(combine_feature)
        combine_feature = combine_feature * visual_feature + combine_feature * language_feature

        predicts = self.fc(combine_feature)
        
        if not self.training:
            predicts = F.softmax(predicts, axis=2)

        return {'predict': predicts, 'visual_out':visual_out}