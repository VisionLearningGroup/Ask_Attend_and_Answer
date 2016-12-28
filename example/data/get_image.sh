### get the coco images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip val2014.zip

wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip test2015.zip

### get the VQA annotations
mkdir Annotations
cd Annotations
wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
unzip Annotations_Train_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip
unzip Annotations_Val_mscoco.zip

### get the VQA Questions
cd ..
mkdir Questions
cd Questions
wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
unzip Questions_Test_mscoco.zip

cd ..
