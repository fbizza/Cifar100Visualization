# cifar-100-visualization
Interactive dashboard that enables some exploration of data from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

![gif](https://github.com/fbizza/Cifar100Visualization/assets/109001290/1f9fc1ec-e8df-4efe-9adb-636374571b10)

# Goals
1. **Dimensionality Reduction:** it utilizes the t-SNE algorithm over the activation layers of a CNN to create a 2D embedding of high-dimensional data
2. **Explanation:** it implements [LIME](https://arxiv.org/abs/1602.04938) to get the explaination of the class predicted by the deep model, it also shows the top-3 class distribution from the softmax layer
3. **Visualization:** for a nice, customizable and responsive user experience the UI is created using Dash

## Details
A Keras model based on the VGG16 architecture is used for the predictions
<p align="center">
  <img src="https://github.com/fbizza/cifar-100-visualization/assets/109001290/b287a405-37de-4fe7-974a-b24a88bf4f63" alt="VGG16" width="600" />
</p>
To get valuable insights about the model (and dataset) weaknesses it is interesting to visualize the explanations of the wrong predictions, users can filter such images using a dropdown menu. 

Examples: 
<p align="center">
  <img width="300" alt="donna" src="https://github.com/fbizza/cifar-100-visualization/assets/109001290/e1a31902-397a-47b4-9b00-c9b54baf2d96">
  <img width="300" alt="donna" src="https://github.com/fbizza/cifar-100-visualization/assets/109001290/4d0ad0bf-0fb0-40f7-a67f-03589b2d1449">
</p>


## Run the code
From the `XAI` folder: 
```
pip install -r requirements.txt
```
Then run `app.py` and just click on the link, it might take a while (~20 seconds) depending on the hardware of your system

