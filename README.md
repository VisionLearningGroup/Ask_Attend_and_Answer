# Ask_Attend_and_Answer
 
Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering


# Code

Instructions for training and testing the "SMem-VQA Two-Hop" model:

1. Download the provided caffe folder and install caffe following the instructions in http://caffe.berkeleyvision.org/installation.html .

2. cd example/data/

   run ./get_image.sh to download MSCOCO images, and VQA annotations and questions.

3. cd example/

   run ./train/train_mm.sh to train the model.

4. Model trained on VQA dataset: [SMem-VQA](https://drive.google.com/file/d/0BxLtQPBFL-uLUFExNEpHNUIyUzQ/view)

5. predict the answers for the images and questions in VQA test-dev dataset 

   python ./prediction/predict_json.py

# Citation

    @article{xu2015ask,
        Author = {Xu, Huijuan and Saenko, Kate},
        Title = {Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering},
        Journal = {arXiv preprint arXiv:1511.05234},
        Year = {2015}
    }
