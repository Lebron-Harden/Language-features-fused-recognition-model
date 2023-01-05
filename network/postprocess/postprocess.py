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
import numpy as np
import paddle


def argsec_max(preds):
    num_class = preds.shape[-1]
    #preds = paddle.to_tensor(preds)
    #preds = paddle.reshape(preds, [-1, num_class])
    
    idx = np.zeros((preds.shape[0]))
    probs = np.zeros((preds.shape[0]))
    for i in range(preds.shape[0]):
        max = -999
        max_i = 0
        probs[i] = -9999
        for j in range(preds.shape[1]):
            if max < preds[i][j]:
                probs[i] = max
                idx[i] = max_i
                max = preds[i][j]
                max_i = j
            elif probs[i] < preds[i][j]:
                probs[i] = preds[i][j]
                idx[i] = j
                          
    return idx, probs


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):

        self.beg_str = "sos"
        self.end_str = "eos"

        
        self.character_str = []
        assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
            character_type)
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = list(self.character_str)

        
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                #a = int(text_index[batch_idx][idx])
                
                char_list.append(self.character[int(text_index[batch_idx][idx])])

                
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 pr_thresh=0.8,
                 cooccurrence_relation_path=None,
                 mode='Train',
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)
        self.pr_thresh = pr_thresh
        self.cooccurrence_relation_path = cooccurrence_relation_path
        self.mode = mode

    def __call__(self, preds, label=None, *args, **kwargs):

        if isinstance(preds, dict):
            pred = preds['predict']
        else:
            pred = preds
        char_num = len(self.character_str) + 2
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, self.max_text_length])
        preds_prob = np.reshape(preds_prob, [-1, self.max_text_length])


        if self.mode == 'Eval':
            preds_idx = self.PredictionRectification(pred, preds_idx, preds_prob, char_num)
        
        
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        label = self.decode(label)

        return text, label #, sec_text, preds_prob, sec_probs

        
    def PredictionRectification(self, pred, preds_idx, preds_prob, char_num):
        relation = np.load(self.cooccurrence_relation_path)
        sec_idx, sec_probs = argsec_max(pred)
        sec_idx = np.reshape(sec_idx, [-1, self.max_text_length])
        sec_probs = np.reshape(sec_probs, [-1, self.max_text_length])

        one_wrong_num = 0
        perplex = []
        for i in range(preds_idx.shape[0]):  
            num = 0
            x = -1
            y = -1
            for j in range(preds_idx.shape[1]):
                if preds_prob[i][j] < self.pr_thresh:
                    num += 1
                    x=i
                    y=j
                if num > 1:
                    break

            if num == 1 :
                a = preds_idx[x][y]
                b = sec_idx[x][y]
                if (a != char_num - 1 and a!= char_num -2) and (b != char_num - 1 and b != char_num - 2):
                    one_wrong_num += 1
                    perplex.append((x,y))
        print("only one char very similar in a label: {}/{}".format(one_wrong_num, preds_idx.shape[0]))
        for i in range(len(perplex)):
            (x,y) = perplex[i]
            cur_pred = preds_idx[x]
            max_char = int(preds_idx[x][y])
            sec_char = int(sec_idx[x][y])
            max_score = 0
            sec_score = 0
            
            for j in range(len(cur_pred)):
                if cur_pred[j] == char_num-1 or cur_pred[j] == char_num-2:
                    continue
                if j == y:
                    continue

                cur_char = int(cur_pred[j])
                dis = abs(j-y)
                max_score += 1.0*relation[max_char][cur_char]/dis
                sec_score += 1.0*relation[sec_char][cur_char]/dis

            if sec_score > max_score:
                preds_idx[x][y] = sec_char
                print("rectify one char!")

        sec_text = self.decode(sec_idx, sec_probs)

        return preds_idx
        

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


