# Ask_Attend_and_Answer
 
Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering


# Code

Instructions for training and testing the "SMem-VQA Two-Hop" model:

1. Download the provided caffe folder and install caffe following the instructions in http://caffe.berkeleyvision.org/installation.html .

2. Download MSCOCO images, and VQA annotations and questions:
  
   cd example/data/

   ./get_image.sh
   
3. Generate the hdf5 data for training and testing:
 
   cd example/

   python ./data/generate_h5_data/generate_h5_data.py

4. Train the model:
 
   cd example/

   run ./train/train_mm.sh 

5. Model trained on VQA dataset: [SMem-VQA](https://drive.google.com/file/d/0BxLtQPBFL-uLUFExNEpHNUIyUzQ/view)

6. Predict the answers for the images and questions in VQA test-dev dataset:
 
   cd example/
   
   python ./prediction/predict_json.py


# Citation

@inproceedings{xu2016ask,
  title={Ask, attend and answer: Exploring question-guided spatial attention for visual question answering},
  author={Xu, Huijuan and Saenko, Kate},
  booktitle={European Conference on Computer Vision},
  pages={451--466},
  year={2016},
  organization={Springer}
}



