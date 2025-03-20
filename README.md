🐶🐱 Dog vs Cat Image Classification

📌 Overview
This project utilizes a Convolutional Neural Network (CNN) to classify images as either a dog or a cat. The dataset is preprocessed and fed into a deep learning model, which is trained to distinguish between the two categories.

🛠️ Technologies Used
Python
TensorFlow & Keras (Deep Learning Framework)
NumPy (Numerical Computation)
Matplotlib (Data Visualization)
PIL (Pillow) (Image Processing)

📂 Dataset
The dataset consists of labeled images stored in separate directories:
🐱 cat → Label 0
🐶 dog → Label 1
Each image is resized to 64x64 pixels and converted into a NumPy array for processing.

🏗️ Model Architecture
The CNN model comprises the following layers:
Convolutional Layers with ReLU activation
Max Pooling Layers for feature extraction
Flatten Layer to convert into a 1D array
Fully Connected (Dense) Layers
Output Layer with Softmax activation for classification

🎯 Training Details
Loss Function: Categorical Cross-Entropy
Optimizer: Adam
Metric: Accuracy
The dataset is used to train the model, allowing it to differentiate between cats and dogs over multiple epochs.

✅ Evaluation & Testing
The trained model is tested on new images to assess its accuracy.
Predictions are displayed alongside images for verification.

🚀 How to Run
1️⃣ Install Required Dependencies
Run the following command in your terminal:

pip install tensorflow keras numpy matplotlib pillow
2️⃣ Run the Jupyter Notebook
Execute the detect_dog_vs_cat.ipynb notebook to train and test the model.

📊 Results
The model successfully classifies images with high accuracy.
Example predictions are displayed in the notebook.

🔍 Future Enhancements
📌 Expand the dataset for better generalization.
📌 Implement Data Augmentation to improve robustness.
📌 Optimize Hyperparameters to boost accuracy.
