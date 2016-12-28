# Ask_Attend_and_Answer
 
Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering


# Code

Instructions for training and testing the "SMem-VQA Two-Hop" model:

1. Download the provided caffe folder and install caffe following the instructions in http://caffe.berkeleyvision.org/installation.html .

2. Download MSCOCO images, and VQA annotations and questions
  
   cd example/data/

   ./get_image.sh
   
3. get the hdf5 data for training and testing
 
   cd example/

   python ./data/generate_h5_data/generate_h5_data.py

4. train the model
 
   cd example/

   run ./train/train_mm.sh 

5. Model trained on VQA dataset: [SMem-VQA](https://drive.google.com/file/d/0BxLtQPBFL-uLUFExNEpHNUIyUzQ/view)

6. predict the answers for the images and questions in VQA test-dev dataset 
 
   cd example/
   
   python ./prediction/predict_json.py


# Citation

    @article{xu2015ask,
        Author = {Xu, Huijuan and Saenko, Kate},
        Title = {Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering},
        Journal = {arXiv preprint arXiv:1511.05234},
        Year = {2015}
    }
