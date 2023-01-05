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


import Levenshtein

class RecMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        accurate_rate = 0.0
        correct_rate = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")

            accurate_rate += Levenshtein.distance(pred, target) / len(target)
            if len(pred) == len(target):
                correct_rate += Levenshtein.distance(pred, target) / len(target)
            else:
                correct_rate += abs((Levenshtein.distance(pred, target) - abs(len(pred) - len(target)))) / len(target)

            if pred == target:
                correct_num += 1
            all_num += 1
        self.all_num += all_num
        self.correct_num += correct_num
        self.accurate_rate += accurate_rate
        self.correct_rate += correct_rate

        return {
            'acc': correct_num / all_num,
            'accurate_rate': 1 - accurate_rate / all_num,
            'correct_rate': 1 - correct_rate / all_num
        }

    def get_metric(self):

        acc = 1.0 * self.correct_num / self.all_num
        accurate_rate = 1 - self.accurate_rate / self.all_num
        correct_rate = 1- self.correct_rate / self.all_num
        self.reset()

        return {
                 'acc': acc,
                 'accurate_rate': accurate_rate,
                 'correct_rate': correct_rate
            }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.accurate_rate = 0
        self.correct_rate = 0
        





'''class RecMetric2(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        if len(pred_label) == 2:
            preds, labels = pred_label
            correct_num = 0
            all_num = 0
            accurate_rate = 0.0
            correct_rate = 0.0

            for (pred, pred_conf), (target, _) in zip(preds, labels):
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

                accurate_rate += Levenshtein.distance(pred, target) / len(target)
                if len(pred) == len(target):
                    correct_rate += Levenshtein.distance(pred, target) / len(target)
                else:
                    correct_rate += abs((Levenshtein.distance(pred, target) - abs(len(pred) - len(target)))) / len(target)

                if pred == target:
                    correct_num += 1

                all_num += 1

            self.all_num += all_num
            self.correct_num += correct_num
            self.accurate_rate += accurate_rate
            self.correct_rate += correct_rate

            return {
                'acc': correct_num / all_num,
                'accurate_rate': 1 - accurate_rate / all_num,
                'correct_rate': 1 - correct_rate / all_num
            }
        
        else:
            preds, labels, sec_preds, pred_probs, sec_probs = pred_label
            correct_num = 0
            all_num = 0
            accurate_rate = 0.0
            correct_rate = 0.0
            
            ii = -1
            for (pred, pred_conf), (target, _) , (sec, sec_prob) in zip(preds, labels, sec_preds):
                ii += 1
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
                
                accurate_rate += Levenshtein.distance(pred, target) / len(target)
                if len(pred) == len(target):
                    correct_rate += Levenshtein.distance(pred, target) / len(target)
                else:
                    correct_rate += abs((Levenshtein.distance(pred, target) - abs(len(pred) - len(target)))) / len(target)

                if pred == target:
                    correct_num += 1
                    print("\n\n---------------************------------------")
                    print(pred)
                    for i in range(len(pred)):
                        if pred_probs[ii][i] < 0.9:
                            print("max prob:" + str(pred_probs[ii][i]) + "  second max prob: " + str(sec_probs[ii][i]))
                            print("max prob char: " + pred[i] + "           second max prob char: " + sec[i])
                    print("---------------************------------------\n\n")

                else:
                    print("\n\n---------------************------------------")
                    prev = pred
                    for i in range(min(len(pred), len(target))):
                        if pred[i] != target[i]:
                            print("\n")
                            print("max probability of current position:" + str(pred_probs[ii][i]))
                            print("second probability of current position:" + str(sec_probs[ii][i]))
                            print("truth: " + target[i] + "  max pred: " + pred[i] + "  second pred: " + sec[i])
                            pred = pred[:i] + sec[i] + pred[i+1:]
                            
                    if pred == target:
                        correct_num += 1
                    print("\nwhen wrong, use second max probability to replace")
                    print("preds      :" + prev)
                    print("rectified  :" + pred)
                    print("groundtruth:" + target)

                    print("---------------************------------------\n\n")
                    
                    
                all_num += 1
            self.all_num += all_num
            self.correct_num += correct_num
            self.accurate_rate += accurate_rate
            self.correct_rate += correct_rate

            return {
                'acc': correct_num / all_num,
                'accurate_rate': 1 - accurate_rate / all_num,
                'correct_rate': 1 - correct_rate / all_num
            }

    def get_metric(self):

        acc = 1.0 * self.correct_num / self.all_num
        accurate_rate = 1 - self.accurate_rate / self.all_num
        correct_rate = 1- self.correct_rate / self.all_num
        self.reset()

        return {
                 'acc': acc,
                 'accurate_rate': accurate_rate,
                 'correct_rate': correct_rate
            }

    def reset(self):
        self.all_num = 0
        self.correct_num = 0
        self.accurate_rate = 0
        self.correct_rate = 0
        
'''