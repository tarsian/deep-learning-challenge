# deep-learning-challenge
<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=header" />

# Alphabet Soup Charity - Deep Learning Model

## Overview of the Analysis

The purpose of this analysis is to create a deep learning model to predict if an Alphabet Soup-funded organization will be successful based on various features in the dataset. The analysis involves preprocessing the data, building and training a neural network model, optimizing the model for better performance, and finally evaluating its effectiveness.

## Instructions

### Step 1: Preprocess the Data

Using Pandas and scikit-learn¡¯s `StandardScaler()`, preprocess the dataset. This step prepares the data for compiling, training, and evaluating the neural network model.

- **Upload the starter file to Google Colab**.
- **Read in the `charity_data.csv` to a Pandas DataFrame**.
- **Identify target and feature variables**.
- **Drop the `EIN` and `NAME` columns**.
- **Determine the number of unique values for each column**.
- **Combine rare categorical variables into a new value, 'Other'**.
- **Encode categorical variables using `pd.get_dummies()`**.
- **Split the data into features (X) and target (y) arrays**.
- **Split the data into training and testing datasets**.
- **Scale the data using `StandardScaler()`**.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, design a neural network to create a binary classification model.

- **Create a neural network model with input features and hidden nodes**.
- **Add the first hidden layer with an appropriate activation function**.
- **Add a second hidden layer if necessary**.
- **Create an output layer with an appropriate activation function**.
- **Compile and train the model**.
- **Evaluate the model using the test data**.
- **Save the model to an HDF5 file (`AlphabetSoupCharity.h5`)**.

### Step 3: Optimize the Model

Optimize the model to achieve a target predictive accuracy higher than 75%.

- **Adjust the input data**:
  - Drop more or fewer columns.
  - Create more bins for rare occurrences in columns.
  - Increase or decrease the number of values for each bin.
- **Modify the model structure**:
  - Add more neurons to a hidden layer.
  - Add more hidden layers.
  - Use different activation functions.
  - Adjust the number of epochs.
- **Save the optimized model to an HDF5 file (`AlphabetSoupCharity_Optimization.h5`)**.

### Step 4: Write a Report on the Neural Network Model

Write a report on the performance of the deep learning model.

#### Data Preprocessing

- **Target variable**: `IS_SUCCESSFUL`
- **Feature variables**: All columns except `IS_SUCCESSFUL`
- **Removed variables**: `EIN`, `NAME`

#### Compiling, Training, and Evaluating the Model

- **Neurons, layers, and activation functions**: Explain choices made for the model.
- **Model performance**: Whether the target performance was achieved.
- **Steps for increasing model performance**: Details of optimization attempts.

#### Summary

Summarize the overall results and provide a recommendation for using a different model to solve the classification problem.

### Step 5: Copy Files Into Your Repository

- **Download Colab notebooks**.
- **Move them into the Deep Learning Challenge directory in the local repository**.
- **Push the added files to GitHub**.

<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=footer" />