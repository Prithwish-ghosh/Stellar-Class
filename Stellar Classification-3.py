#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset from CSV
df = pd.read_csv('/Users/prithwishghosh/Downloads/star_classification.csv')


# In[37]:


df


# In[31]:


# Create subsets based on class values
galaxy_subset = df[df['class'] == 'GALAXY']
quasar_subset = df[df['class'] == 'QSO']
star_subset = df[df['class'] == 'STAR']

# Print the subsets

galaxy_subset.head()


# In[11]:


quasar_subset.head()


# In[12]:


star_subset.head()


# In[33]:


# Plot histograms for each subset
plt.hist(galaxy_subset['redshift'], bins=10, color='blue', alpha=0.5, label='Galaxy')
plt.hist(quasar_subset['redshift'], bins=10, color='green', alpha=0.5, label='Quasar')
plt.hist(star_subset['redshift'], bins=10, color='red', alpha=0.5, label='Star')

# Add labels and legend
plt.xlabel('Log Redshift')
plt.ylabel('Frequency')
plt.title('Redshift Histogram for Different Classes')
plt.legend()

# Show plot
plt.show()


# In[36]:


# Plot histograms for each subset
plt.hist(star_subset['redshift'], bins=10, color='red', alpha=0.5, label='Star')

# Add labels and legend
plt.xlabel('Log Redshift')
plt.ylabel('Frequency')
plt.title('Redshift Histogram for Different Classes')
plt.legend()

# Show plot
plt.show()


# In[47]:


df


# In[22]:


df.shape


# In[38]:


import pandas as pd

# Assuming 'data' is your DataFrame containing the dataset
# Replace 'data' with the name of your DataFrame if it's different

selected_columns = df.iloc[:, 3:8]  # Columns 4 to 8 (indexing starts from 0)
data = pd.concat([selected_columns, df.iloc[:, 13:15]], axis=1)  # Concatenate columns 14 and 15


# In[39]:


data.head()


# In[43]:


# Assuming 'class' column is categorical, you may need to encode it if it's not already numeric
# You can use LabelEncoder or get_dummies depending on your data
# Example if 'class' is not numeric:
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# data['class'] = label_encoder.fit_transform(data['class'])

# Split dataset into features (X) and target (y)
X = data[['redshift']]  # Features
y = data['class']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network classification
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_preds)
print("Neural Network Accuracy:", nn_accuracy)


# In[44]:


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Adaptive Decision Learner (ADL) model
class ADL:
    def __init__(self, base_classifier, n_iterations):
        self.base_classifier = base_classifier
        self.n_iterations = n_iterations
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            model = self.base_classifier()
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return np.mean(predictions, axis=1)

# Create and train the ADL model
adl = ADL(base_classifier=DecisionTreeClassifier, n_iterations=10)
adl.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred = adl.predict(X_test_scaled)

# Convert predicted probabilities to class labels
y_pred_binary = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("ADL Accuracy:", accuracy)


# In[45]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': 3,  # Number of classes
    'eval_metric': 'mlogloss',  # Metric to optimize
    'max_depth': 6,  # Maximum depth of the tree
    'eta': 0.3,  # Learning rate
    'subsample': 0.8,  # Percentage of samples used in each iteration
    'colsample_bytree': 0.8,  # Percentage of features used in each iteration
    'alpha': 10,  # L1 regularization term
    'lambda': 1  # L2 regularization term
}

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
y_pred = bst.predict(dtest)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[46]:


import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define AdaBoost classifier
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model
adaboost_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[47]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], 1)),  # LSTM layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape input data for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test_reshaped, y_test)[1]
print("Accuracy:", accuracy)


# In[49]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[50]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target variable (y)
X = data[['redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])  # Adding an extra dimension for the time steps

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[51]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target variable (y)
X = data[[ 'redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with GRU
X = X.reshape(X.shape[0], 1, X.shape[1])  # Adding an extra dimension for the time steps

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # GRU layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[53]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = data[['z', 'redshift']].values
y = data['class'].values
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_rf = rf_clf.predict(X_test_scaled)

# Calculate accuracy for RandomForestClassifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForestClassifier Accuracy:", accuracy_rf)

# Define and train SVC
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_svm = svm_clf.predict(X_test_scaled)

# Calculate accuracy for SVC
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVC Accuracy:", accuracy_svm)

# Define and train LogisticRegression
lr_clf = LogisticRegression(solver='liblinear', random_state=42)
lr_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_lr = lr_clf.predict(X_test_scaled)

# Calculate accuracy for LogisticRegression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("LogisticRegression Accuracy:", accuracy_lr)


# In[46]:


import seaborn as sns
# Define colors used as colorcodes
blue = '#51C4D3' # To mark drinkable water
green = '#74C365' # To mark undrinkable water
red = '#CD6155' # For further markings
orange = '#DC7633' # For further markings

# Plot the colors as a palplot
sns.palplot([blue])
sns.palplot([green])
sns.palplot([red])
sns.palplot([orange])


# In[49]:


# Clear matplotlib and set style 
plt.clf()
plt.style.use('ggplot')

# Create subplot and pie chart
fig1, ax1 = plt.subplots()
ax1.pie(data['class'].value_counts(), colors=[green, blue, red], labels=['Galaxy', 'Star', 'Quasar'], autopct='%1.1f%%', startangle=0, rotatelabels=False)

#draw circle
centre_circle = plt.Circle((0,0),0.80, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  

# Set tighten layout and show plot 
plt.tight_layout()
plt.show()


# In[54]:


import matplotlib.pyplot as plt

# Data for the accuracy table
machine_learning_algorithms = [
    "XGBoost", "CNN", "RNN", "AdaBoost", 
    "Adaptive Decision Learner", "LSTM Networks", 
    "GRU", "Random Forest Classifier", 
    "SVM", "Logistic Regression"
]
accuracy_scores = [
    94.71, 91.54, 94.60, 80.40, 
    91.62, 94.6, 93.46, 
    94.71, 94.35, 93
]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(machine_learning_algorithms, accuracy_scores, color='skyblue')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Machine Learning Algorithms with respect to Redshift only')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[60]:


import matplotlib.pyplot as plt

# Data for the accuracy table
machine_learning_algorithms = [
    "XGBoost", "CNN", "RNN", "AdaBoost", 
    "Adaptive Decision Learner", "LSTM Networks", 
    "GRU", "Random Forest Classifier", 
    "SVM", "Logistic Regression"
]
accuracy_scores_redshift = [
    94.71, 91.54, 94.60, 80.40, 
    91.62, 94.6, 93.46, 
    94.71, 94.35, 93
]
accuracy_scores = [
    97.395, 95.87, 96.90, 80.48, 
    96.635, 96.46, 96.609, 
    97.805, 96.11, 93.465
]

# Plotting the bar plot for redshift only
plt.figure(figsize=(10, 6))
plt.bar(machine_learning_algorithms, accuracy_scores, color='red', label='All Parameters', alpha=0.7)
plt.bar(machine_learning_algorithms, accuracy_scores_redshift, color='black', label='Redshift Only')

plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Accuracy with Redshift Only vs. All Parameters')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[52]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Separate features (X) and target variable (y)
X = data[['u', 'g', 'r', 'i', 'z', 'redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Extract features using CNN
train_features = cnn_model.predict(X_train)
test_features = cnn_model.predict(X_test)

# Define SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(train_features, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[54]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from sklearn.metrics import accuracy_score

# Separate features (X) and target variable (y)
X = data[['u', 'g', 'r', 'i', 'z', 'redshift']].values
y = data['class'].values

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP model
mlp_input = Input(shape=(X_train.shape[1],))
mlp_dense1 = Dense(64, activation='relu')(mlp_input)
mlp_dense2 = Dense(32, activation='relu')(mlp_dense1)

# Define CNN model
cnn_input = Input(shape=(X_train.shape[1], 1))
cnn_conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
cnn_pool1 = MaxPooling1D(pool_size=2)(cnn_conv1)
cnn_flatten = Flatten()(cnn_pool1)

# Combine MLP and CNN
combined = concatenate([mlp_dense2, cnn_flatten])

# Dense layer for classification
output = Dense(3, activation='softmax')(combined)

# Create model
model = Model(inputs=[mlp_input, cnn_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train, X_train], y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate([X_test, X_test], y_test)[1]
print("Accuracy:", accuracy)

