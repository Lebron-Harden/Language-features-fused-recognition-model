# Language-features-fused-recognition-model

Introduction
---

This is the model of paper "A language features-fused model for Chinese historical document recognition".

This project is modified on the basis of PaddleOCR. We used the overall framework of PaddleOCR and modified the core part to realize our idea. The link to PaddleOCR is https://github.com/PaddlePaddle/PaddleOCR.

Training
---
Firstly, download two datasets on the website below:

MTHv2:  https://github.com/HCIILAB/MTHv2_Datasets_Release

CLFBL-MY: https://github.com/Lebron-Harden/CLFBL-MY-A-Chinese-historical-document-dataset  



Secondly, download cooccurrence relation matrix which is used in the prediction rectification module: https://drive.google.com/file/d/1mXyX52PqGXn7C52YOjKudWbu-TrJ4UTJ/view?usp=sharing. Then put it in the weights folder. Note that this matrix won't be used during training. 

Thirdly, run [tools\train.py](tools\train.py) to starting training. You can modify some hyper-parameters in [configs\res18_fuse.yml](configs\res18_fuse.yml) before training.



Evaluation
---
Run [tools\eval.py](tools\eval.py) to perform batch evaluation. You can also run [tools\infer.py](tools\infer.py) to recognize a single image.
