# Deep-fake-detection

# **Project Overview**

In the age of advanced image manipulation techniques, distinguishing between authentic and fabricated images is becoming increasingly challenging. 
This project addresses this problem by developing a robust image classification system. The core idea is to leverage the power of pre-trained CNNs (VGG16, ResNet50) to extract rich image features and then employ a Modified Grasshopper Optimization (MGO) algorithm to select the most relevant features, reducing dimensionality and potentially improving model performance and training time. 

# **Features**

Image Preprocessing: Includes resizing, color conversion, Gaussian blur, and normalization.

Data Augmentation: Expands the dataset and improves model generalization through techniques like rotation, shifting, and flipping.

Feature Extraction: Utilizes pre-trained VGG16 and ResNet50 models to extract deep image features.

Modified Grasshopper Optimization (MGO): A custom feature selection algorithm to identify the most discriminative features.

1D CNN Classifier: A convolutional neural network specifically designed to work with the flattened feature vectors.

Pipeline-based Workflow: The project is structured into logical pipelines for preprocessing, feature extraction, MGO feature selection, and CNN classification, making it modular and easy to understand.

Model Evaluation: Includes plotting training history (accuracy and loss) and a confusion matrix to assess model performance.

New Image Prediction: Allows uploading a new image to get a real/fake prediction.

# **Methodology**
The project follows a multi-stage pipeline:

Data Loading and Preprocessing: Real and fake images are loaded, preprocessed (resized, smoothed, normalized), and augmented to increase the dataset size and variability.

Feature Extraction: Pre-trained VGG16 or ResNet50 models are used to extract feature vectors from the preprocessed images. The top classification layers are removed to obtain bottleneck features.

Modified Grasshopper Optimization (MGO) Feature Selection: The MGO algorithm is applied to the extracted feature vectors. The fitness function for MGO is based on the classification accuracy of a simple neural network trained on the selected features, guiding the algorithm to find the subset of features that yields the best classification performance.

1D CNN Classification: A 1D Convolutional Neural Network is built and trained on the features selected by the MGO algorithm.

Evaluation and Prediction: The trained CNN is evaluated on a test set using accuracy and a confusion matrix. The pipeline also includes functionality to predict whether a newly uploaded image is real or fake.

Dataset
The project expects the dataset to be organized into two folders: real2 and fake2, containing the real and fake images respectively. These folders should be placed in your Google Drive and the paths specified in the notebook.

REAL_PATH = '/content/drive/MyDrive/real2'
FAKE_PATH = '/content/drive/MyDrive/fake2'
Note: Ensure you have sufficient images in these folders for training and testing. The current code assumes a binary classification problem.

Getting Started
Prerequisites
Google Colab environment
Access to Google Drive for storing data and outputs
Python 3.7+
Required Python libraries (listed in the code cells)
Installation
This project is designed to be run in Google Colab. No separate installation is required beyond having a Google account and access to Colab.
The necessary libraries are imported within the notebook cells.

Ensure you have the dataset (real and fake images) uploaded to your Google Drive in the specified folder structure (real2 and fake2 within a location like MyDrive).

# **Running the Pipeline**

The project is structured as a series of interconnected pipelines. Execute the code cells in the notebook sequentially to run the entire workflow:

Mount Google Drive: Run the cell to mount your Google Drive.

Define Paths: Run the cells to define the paths for your real and fake image folders and the output directory.

Create Output Directory: Run the cell to create the output directory in your Google Drive.

Define Helper Functions: Run the cells containing helper functions for displaying images, loading images, preprocessing, and augmentation.

Run Preprocessing Pipeline: Execute the preprocess_pipeline() function. This will load, preprocess, augment, and save the data to preprocessed_data.pkl.

Define Feature Extraction Functions: Run the cells defining VGG16 and ResNet50 feature extraction functions.

Run VGG16 Feature Extraction Pipeline: Execute the feature_extraction_pipeline() function to extract and save VGG16 features to extracted_features.pkl.

Run ResNet50 Feature Extraction Pipeline (Optional): Execute the resnet50_feature_extraction_pipeline() function to extract and save ResNet50 features to extracted_features_resnet50.pkl. You can choose to work with either VGG16 or ResNet50 features for the subsequent steps.

Define MGO Feature Selection Functions: Run the cells defining the MGO algorithm and related plotting functions.

Run MGO Feature Selection Pipeline (VGG16 or ResNet50): Execute the mgo_feature_selection_pipeline() (for VGG16 features) or mgo_feature_selection_pipeline_resnet50() (for ResNet50 features) function to perform feature selection and save the results.

Define CNN Classification Functions: Run the cells defining the CNN model building, training, plotting, confusion matrix, and new image processing functions.

Run CNN Classification Pipeline: Execute the cnn_classification_pipeline() function. This will load the selected features, train the CNN, evaluate it, save the model, and prompt you to upload an image for prediction.

Follow the prompts in the output to upload an image for classification after the CNN is trained.
