# Binary classification in JAX using an MLP

##Author:  Omar Shalaby



###Relevant Links:

https://github.com/oms9/CS301-Group3-Project

https://www.kaggle.com/competitions/breast-cancer-data/overview

https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset



---
### 1. Abstract
This project aims to classify breast tumors using the Wisconsin Breast Cancer Diagnostic dataset into two categories. Benign and Malignant, based on the features of the tumor.
<br><br>
The dataset has 569 entries:
<ul type = "circle">
<li>Benign: 357
<li>Malignant: 212
</ul>


<br>

I am going to tackle this problem by creating an MLP (multi-layer perceptron) in JAX with the help of Haiku, a neural network building library for JAX, which is a python library that allows for tasks to be run asynchronously on accelerator hardware and performs matrix manipulation and differentiation on a GPU/TPU.

This dataset is also present in scikit learn so I will be using that module since it is already prepared for importing

This MLP will perform binary classification and will be splitting the data into 80-20 percents, 80% for training and 20% for validation.
<br>

Results: The results of this project is an MLP capable of classifying breast tumors based on metrics describing the tumor with 96% test accuracy in just 500 epochs, taking only about 10 minutes to train.

<br>

---
### 2. Introduction
Breast cancer is the single most common cancer among women, it is a large threat and an even larger cause of death for women worldwide. Approximately 124 out of 100,000 women are diagnosed with breast cancer, 23 of which are likely to die because of the tumor.

<br>

Early detection is key in preventing death, if detected early there is chance for successful treatment, 30% and steadily rising. So it is important that we spend a lot of effort on early detection so that the research and effort dedicated towards treatment is more impactful.

<br>

As we can see from the results, AI can help doctors and therefore women around the world by analyzing every bit of data about the detected tumors and providing accurate (up to 96% with little training!) predictions of the type of tumor.

<br>

---


### 3. Related works
[This](https://www.kaggle.com/code/ratnesh88/breast-cancer-prediction-using-pytorch/notebook) notebook served as a guide to help tackle this problem and [this](https://theaisummer.com/jax-transformer/) article helped me understand how to use haiku to build an MLP.

<br>

The first notebook is different from my approach because it was written in pytorch and took 500 epochs to produce good results.

<br>

I plan to tackle this problem in JAX, with the help of Haiku but the problem is still the same, a binary classification problem.

<br>

---


### 4. Data
The data I am working with for this project is numerical and it takes the form of a
set of paramters 30 in total, describing the tumor. These parameters are:

```
radius (mean)
texture (mean)
perimeter (mean)
area (mean)
smoothness (mean)
compactness (mean)
concavity (mean)
concave points (mean)
symmetry (mean)
fractal dimension (mean)
radius (standard error)
texture (standard error)
perimeter (standard error)
area (standard error)
smoothness (standard error)
compactness (standard error)
concavity (standard error)
concave points (standard error)
symmetry (standard error)
fractal dimension (standard error)
radius (worst)
texture (worst)
perimeter (worst)
area (worst)
smoothness (worst)
compactness (worst)
concavity (worst)
concave points (worst)
symmetry (worst)
fractal dimension (worst)
```

<br>

The data came from: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)


<br>

As mentioned, the data has 569 entries, giving us a total of 17,070 data points. 

Preprocessing the data was not neccessary aside from splitting it 80-20 into 80% training and 20% validation categories and a simple normalization procedure.

<br>

---


### 5. Methods
My approach is using a multi-layer perceptron to perform binary classification using the datapoints of a given tumor.

This is a good approach because perceptrons by nature are exceptional at recognizing patterns, even discovering patterns that human trainers and supervisors did not even know existed in the data.

This is more useful than simple binary classification using a binary tree or trying to linearly separate the data because it is more flexible and pattern driven rather than being a threshhold game where 

```
if  numbers  >= x: class a
else: class b
```

<br>

---


### 6. Experiments
To demonstrate that my approach solves the problem, I evaluted the model after training it to see the accuracy of it's predictions.

<br>

We can use TensorFlow and PyTourch to compare performance to the pure JAX implementation of the perceptron and evalute the performance and the speed of the model.

<br>

I will visualize the weights to try to understand which paramters are most impactful on the decision/classification problem when it comes to the tumors.

<br>

---


### 7. Conclusion
We can see from the evaluations and results that the perceptron was very successful!

<br>

To further improve this model, we can try to scavenge more data from more countries and different body types and ethnicities to try and strengthen the model or we can instead have different models for different ethnicities as the similarities and differences between the tumors are discovered using the model and the new data points and sources.

<br>

