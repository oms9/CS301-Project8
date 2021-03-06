# Binary classification in JAX using an MLP [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oms9/CS301-Project8/blob/main/CS301_Project_8_colab.ipynb)


Author:  Omar Shalaby

Relevant Links:

https://pantelis.github.io/data-science/aiml-common/lectures/classification/classification-intro/_index.html

https://pantelis.github.io/data-science/aiml-common/lectures/classification/perceptron/_index.html

https://www.kaggle.com/competitions/breast-cancer-data/overview

https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset

https://coderzcolumn.com/tutorials/artificial-intelligence/create-neural-networks-using-high-level-jax-api

Colab notebook for this project:
https://colab.research.google.com/github/oms9/CS301-Project8/blob/main/CS301_Project_8.ipynb

---
## Project report:

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

I am going to tackle this problem by creating an MLP in JAX with the help of Haiku.

This dataset is also present in scikit learn so I will be using that module since it is already prepared for importing

This MLP will perform binary classification and will be splitting the data into 80-20 percents, 80% for training and 20% for validation.
<br>

Results: The results of this project is an MLP capable of classifying breast tumors based on metrics describing the tumor with 96% test accuracy in just 500 epochs, taking only about 10 minutes to train.

---
### 2. Introduction

#### 1. What is an MLP?

> MLP is short for multi-layer perceptron. A fully connected neural network with multiple layers utilizing a feedforward architecture. All the neurons in one layer are connected to the next and each have weights and biases for these connections.



#### 2. What kind of problem is this?

> This is a Binary classification problem. Meaning that we are trying to decide, based on the data and parameters that we have for a certain element whether it belongs to class A or class B, in this case we are trying to decide if a certain tumor is either Malignant or Benign using its metrics.



#### 3. What are JAX and Haiku?

> JAX is a python library that allows tasks to be run asynchronously on accelerator hardware and performs matrix manipulation and differentiation on a GPU/TPU extremely quickly and efficiently, it is designed from the ground up to be a high-performance library for ML research, it is quite similar to Numpy.
> 
> Haiku is a library built on top of JAX that enables users to use familiar object-oriented programming models while allowing full access to JAX???s pure function transformations, Haiku is desgined to make managing model parameters and state simple and easy.



#### 4. Why breast cancer?

> Breast cancer is the single most common cancer among women, it is a large threat and an even larger cause of death for women worldwide. Approximately 124 out of 100,000 women are diagnosed with breast cancer, 23 of which are likely to die because of the tumor.
> 
> Early detection is key in preventing death, if detected early there is chance for successful treatment, 30% and steadily rising. So it is important that we spend a lot of effort on early detection so that the research and effort dedicated towards treatment is more impactful.
> 
> As we can see from the results, AI can help doctors and therefore women around the world by analyzing every bit of data about the detected tumors and providing accurate (up to 96% with little training!) predictions of the type of tumor.

<br>

---

### 3. Related works
[This](https://www.kaggle.com/code/ratnesh88/breast-cancer-prediction-using-pytorch/notebook) notebook served as a guide to help tackle this problem and [this](https://theaisummer.com/jax-transformer/) article helped me understand how to use Haiku to build an MLP, [this](https://coderzcolumn.com/tutorials/artificial-intelligence/create-neural-networks-using-high-level-jax-api) article was vital to understanding how to handle the data while using JAX.


The first notebook is different from my approach because it was written in pytorch and took 500 epochs to produce good results.


I plan to tackle this problem in JAX, with the help of Haiku but the problem is still the same, a binary classification problem.


---

### 4. Data
The data I am working with for this project is numerical and it takes the form of a
set of paramters, 30 in total, to describe the tumor. These parameters are:

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

As mentioned, the data has 569 entries, giving us a total of 17,070 data points. 

Preprocessing the data was not neccessary aside from splitting it 80-20 into 80% training and 20% validation categories and a simple normalization procedure.



---


### 5. Methods
My approach is using a multi-layer perceptron to perform binary classification using the datapoints of a given tumor.

This is a good approach because perceptrons by nature are exceptional at recognizing patterns, even discovering patterns that human trainers and supervisors did not even know existed in the data.

This is more useful than simple binary classification using a binary tree or trying to linearly separate the data because it is more flexible and pattern driven rather than being a threshhold game where:

```
if  numbers  >= x: class a
else: class b
```



---


### 6. Experiments
To demonstrate that my approach solves the problem, I evaluted the model after training it to see the accuracy of it's predictions, over approximately 5 minutes to run the entire script including loading the data and 4.3 minutes of training, the model was able to achieve 96% accuracy for the validation portion of the data.

<br>

We can use TensorFlow and PyTourch to compare performance to the pure JAX implementation of the perceptron and evalute the performance and the speed of the model. (TBD)

<br>

I will visualize the weights to try to understand which paramters are most impactful on the decision/classification problem when it comes to the tumors.

---
### 7. Conclusion
We can see from the evaluations and results that the perceptron was very successful!

To further improve this model, we can try to scavenge more data from more countries and different body types and ethnicities to try and strengthen the model or we can instead have different models for different ethnicities as the similarities and differences between the tumors are discovered using the model and the new data points and sources.

---

## The Implementation:
---

```bash
#Run only once per runtime to install JAX and Haiku
#To see the output of this cell, comment out %%capture but be warned that it is long and a little pointless
%%capture 
!pip install JAX #Install JAX
!pip install dm-haiku #Install Haiku, a neural network building library for JAX.
```
```python
#Importing JAX, Haiku and the data from sklearn.
import jax
import haiku as hk
import jax.numpy as jnp
from sklearn import datasets
from jax import value_and_grad
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Initializing JAX.
key = jax.random.PRNGKey(2) #Setting a random key for reproducability of code.
print('JAX is running on the:', jax.lib.xla_bridge.get_backend().platform) #Checking to see if google provided a GPU to run this notebook.
#Make sure this reads: "JAX is running on the: gpu" during testing.
#           ??????
```
---
### Converting the data from numpy to JAX arrays.

The data as imported form SK learn is not handled on the GPU, so we convert it to be JAX compatible after performing the [80 - 20] split for the [train - test] sets.

---
```python
#Importing the data.
X, Y = datasets.load_breast_cancer(return_X_y=True)

# X represents the features/parameters
# Y represents the prediction. 0 | 1
#      0 ??? Benign, 1 ??? Malignant

#Applying the 80-20 split and randomly shuffling the data using the key: 123
parameters_train, parameters_test, predictions_train, predictions_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=123)

#Converting the data to be JAX compatible, changing from NumPy to JAX arrays instead (which are on the GPU).
parameters_train = jnp.array(parameters_train, dtype=jnp.float32) 
parameters_test = jnp.array(parameters_test, dtype=jnp.float32)
predictions_train = jnp.array(predictions_train, dtype=jnp.float32)
predictions_test = jnp.array(predictions_test, dtype=jnp.float32)
```
---
### Data normalization

Standard data normalization procedure using the ?? (mean) and the ?? (standard deviation).

---
```python
#Find ?? and ??.
mean = parameters_train.mean(axis=0)
std = parameters_train.std(axis=0)

#Normalize the data by subtracting mean from each element and dividing by standard deviation.
parameters_train = (parameters_train - mean) / std
parameters_test = (parameters_test - mean) / std
```
---
### The MLP

Here we define the forward function for the MLP and then we transform it using haiku.

Transformation is vital because it turns the modules/functions to pure JAX functions. This is unique to JAX since it is running on accelerated hardware, pure functions are functions that have the same output for the same input, without any print statements for example.

The MLP function's default parameters are as follows:
```python
MLP(output_size=None,
w_init=None,
b_init=None,
with_bias=True,
activation=jax.nn.relu,
activate_final=False, name=None)
```
We only make the necessary changes, which is shaping the output and keep the default activation function (**ReLU**)


---

```python
def FeedForward(x): #Define the feed forward function.
    mlp = hk.nets.MLP(output_sizes=[5,10,15,1]) #This is the sequence of our layer sizes, 30(to match # of params) ??? 10 ??? 15 ??? 1(output layer)
    return mlp(x) #Returns the output.

model = hk.transform(FeedForward) #Transform the function after we are done into a pure JAX function.
```
---
### The Loss Function

This next block is responsible for defining a loss function to use in training our perceptron.

This is an implementation of the Negative Log Loss function, which takes the form:

```
NegLog(Y, Y`) = 1/n * ( -Y * log(Y') - (1-Y) * log(1-Y'))
```

The function should accept the weights, params and diagnosis(actual) and then apply them to the model and return the loss of the predictions.

---
```python
def NegLogLoss(weights, params, actual):#Dfine the loss function (negative log).
    preds = model.apply(weights, key, params) #This applies the model (only possible after transformation to pure fx).
    preds = preds.squeeze() #Removes axes to achieve a better "fit".
    preds = jax.nn.sigmoid(preds) #Sigmoid activation function call.
    return (- actual * jnp.log(preds) - (1 - actual) * jnp.log(1 - preds)).mean() #The formula for the negtative log function.
```
---
### The Weights updater

This is a very simple function to just update the weights using the learning rate as part of our training loop.

---
```python
def UpdateWeights(weights, gradients):
    return (weights - learning_rate * gradients) #Very simple update step.
```
---
### Training/Model parameters

Speaking of the paramters, let's define them now!

---
```python
params = model.init(key, parameters_train[:5]) #Since parameters are not stored with the model, we have to init them using this function providing an RNG key andsome dummy inputs
epochs = 500 #500 epochs to match the notebook mentioned in 3. Realted Works.
batch_size = 32 #arbitrary batch size number.
learning_rate = jnp.array(0.001) #arbitrary learning rate
#(although it is a little high so Google doesn't suspend the runtime.)
```
---
### The Training loop

Speaking of training, let's train the model!

---
```python
for i in range(1, epochs+1):
    batches = jnp.arange((parameters_train.shape[0]//batch_size)+1) #Indexing the batches.

    losses = [] #We use a list to keep a history of our losses for every state.
    for batch in batches:
        if batch != batches[-1]: start, end = int(batch*batch_size), int(batch*batch_size+batch_size) #Last iteration condition.
        else: start, end = int(batch*batch_size), None

        #This is what one batch looks like.
        X_batch, Y_batch = parameters_train[start:end], predictions_train[start:end]

        #Calling our loss function and updating the parameters while utilizing JAX's value and grad functions to accelerate finding the gradients.
        loss, param_grads = value_and_grad(NegLogLoss)(params, X_batch, Y_batch)
        params = jax.tree_map(UpdateWeights, params, param_grads)
        losses.append(loss)#Append loss, and then loop to next batch.

        #A status print out with the current loss in the loop.
        #I chose to have it re-use the same line so that the output doesn't get messy
        #Besides, we are using a list (losses) to keep track of the loss over epoch anyway.
        print("\rEpoch:", i, "Loss:", loss, end="")
```
---
### Making Predictions

Now to define the predictions function, it is a similar structure to the training loop so we can make predictions in batches to more efficiently compute them over the dataset.

---
```python
def MakePredictions(weights, params, batch_size=32):
    batches = jnp.arange((params.shape[0]//batch_size)+1) #Indexing the batches, again.

    predictions = []#Same technique as keeping history of losses.
    for batch in batches:
        if batch != batches[-1]: start, end = int(batch*batch_size), int(batch*batch_size+batch_size) #Last iteration condition.
        else: start, end = int(batch*batch_size), None 
        parameters_batch = params[start:end] #Preparing the list
        predictions.append(model.apply(weights, key, parameters_batch))#Apply the model on the data and log the predicitons.

    return predictions
```
```python
#Now to make predictions using the training set.
output_predictions = MakePredictions(params, parameters_train, 32) #Running the make predictions function defined here ???
output_predictions = jnp.concatenate(output_predictions).squeeze() #Shaping the predictions.
output_predictions = jax.nn.sigmoid(output_predictions) #Sigmoid activation function to make this next step easier.
output_predictions = (output_predictions > 0.5).astype(jnp.float32) #The outputs are in the range from 0 ??? 1, we classify them using 0.5 as a threshhold.
```
```python
#Now to prepare the validation predictions, same procedure as the output_predictions above ???
validation_predictions = MakePredictions(params, parameters_test, 32)
validation_predictions = jnp.concatenate(validation_predictions).squeeze()
validation_predictions = jax.nn.sigmoid(validation_predictions)
validation_predictions = (validation_predictions > 0.5).astype(jnp.float32)
```
---
### Performance evaluation.

Now we are going to see the score and then the accuracy of the model.

We format the output because the loss function's raw output can be a bit messy.

---
```python
#Scores:
print("Test  NegLogLoss Score : {:.2f}".format(NegLogLoss(params, parameters_test, predictions_test)))
print("Train NegLogLoss Score : {:.2f}".format(NegLogLoss(params, parameters_train, predictions_train)))
print()
#Accuracy evaluation:
print("Train Accuracy : {:.2f}".format(accuracy_score(predictions_train, output_predictions)))
print("Test  Accuracy : {:.2f}".format(accuracy_score(predictions_test, validation_predictions)))
```
