# Cell_Neuclei_Detection_via_Semantic_Segmentation
 
## 1. Summary
<p>The objective of this project is to train a model that is able to detect cell nuclei from biomedical images. Due to various shapes and sizes in the nuclei, semantic segmentation is suitable to be selected to detect them. <br>

The model is trained with [Data Science Bowl 2018 dataset](https://www.kaggle.com/c/data-science-bowl-2018). </p>

## 2. IDE and Framework
<p>The project is built with Spyder as the main IDE. The main frameworks used in this project are TensorFlow, Numpy, Matplotlib, OpenCV and Scikit-learn.</p>

## 3. Methodology

<p>The methodology is inspired by a documentation available in the official TensorFlow website (https://www.tensorflow.org/tutorials/images/segmentation).
</p>

### 3.1 Input Pipeline
<p>The dataset files contains a train folder for training data and test folder for testing data, in the format of images for inputs and image masks for the labels. The input images are preprocessed with feature scaling. The labels are preprocessed such that the values are in binary of 0 and 1. No data augmentation is applied for the dataset. The train data is split into train-validation sets, with a ratio of 80:20</p>

### 3.2 Model Pipeline
<p>The model architecture used for this project is U-Net. Do refer to the TensorFlow documentation for further details. To summarize, the model consist of two components, the downward stack, which serves as the feature extractor, and upward stack, which helps to produce pixel-wise output.<br>
 
The model is trained with batch size 16 and 100 epochs. The training stops at epochs of 24 after early stopping is applied with training accuracy of 96% and loss 0.0914.</p>

![model_p4](https://user-images.githubusercontent.com/72061179/165106550-bab0d142-ea5c-4aea-be1f-ba49117acdd0.png)


## 4. Result

<p>Some predictions made with model using test data shown below.</p>

![img1](https://user-images.githubusercontent.com/72061179/164994404-419c980e-5607-43fa-9bc7-0783896c77f4.png)

![img2](https://user-images.githubusercontent.com/72061179/164994410-beca0e08-6a91-4b29-b469-47c311e293ab.png)

<p>The evaluated model with test data is shown below.</p>

![result](https://user-images.githubusercontent.com/72061179/164993806-514ea380-d737-4401-98df-c4d3aebaa734.png)
