# FECNet
A Keras implementation of FECNet, which proposed in "A Compact Embedding for Facial Expression Similarity"



The original paper is ["A Compact Embedding for Facial Expression Similarity"](https://arxiv.org/abs/1811.11283v)



I implemented the original structure with Keras 2.2.0

Using Tensorflow1.3.0 backend



Training device: GTX 1060 6G



Due to the limitation of device, I didn't train the model with whole FEC dataset.



Dataset link: [FEC](https://ai.google/tools/
datasets/google-facial-expression/.)


## Step one
Download the FEC dataset to the data/

## Step two
Run the image_extract.py frist, then run the export_train_label
(If you are in China, a VPN is necessary. I strongly suggest you rent a oversea server to run the code and download those images, this will save you a lot of time.)

## Step three
Run the FEC.py to train a model, or you can use the create_model method in you own training code. Like in Classifi2.py.




I think there may have some problem in triplet loss part or read triplet images part, if you find it, please tell me, thanks.



If you like this project, offer me a star!

