Overview
This project implements time series forecasting of CPU usage metrics, including min_cpu, max_cpu, and avg_cpu, using LSTM, GRU, and Independent RNN models. The goal is to predict these metrics based on historical data from an azure.csv dataset. The project uses Python and TensorFlow to build, train, and evaluate the models.

Dataset
The dataset, azure.csv, contains the following:

timestamp: The time of each observation.
min_cpu: Minimum CPU usage.
max_cpu: Maximum CPU usage.
avg_cpu: Average CPU usage.
Data Preprocessing
Loading and Visualizing:

Data is loaded using pandas and plotted to observe trends.
Timestamps are converted to datetime format, and the timestamp column is set as the index.
Splitting:

Data is split into 80% training and 20% testing datasets.
Scaling:

The data is normalized to the range [0, 1] using MinMaxScaler.
Sequence Generation:

A generator function creates input-output sequences for the models, with a configurable time lag (TIME_STEPS).
Models
1. LSTM
Consists of:
Two LSTM layers (512 units each).
A dense layer for output (3 units for min_cpu, max_cpu, avg_cpu).
Optimizer: Adam
Loss Function: Mean Absolute Error (MAE).
2. GRU
Same architecture as the LSTM model, replacing LSTM layers with GRU layers.
3. Independent RNN
Similar structure, replacing LSTM/GRU layers with SimpleRNN layers.
Training
Configuration
Batch Sizes: Trained on batch sizes of 64, 128, and 256.
Epochs: 10
Validation Split: 25% of the training data.
Callbacks:
Early Stopping to prevent overfitting.
Learning Rate Reduction for better convergence.
Performance Metrics
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
MAPE (Mean Absolute Percentage Error)
Evaluation
Predictions
The test data is predicted, scaled back to the original range, and plotted:
max_cpu: Actual vs. Predicted
min_cpu: Actual vs. Predicted
avg_cpu: Actual vs. Predicted
Results
Performance metrics calculated include:

RMSE: Measures the standard deviation of the prediction errors.
MAE: Measures the average magnitude of the errors.
MAPE: Measures the percentage error.
Dependencies
The following Python libraries are required:

numpy
pandas
matplotlib
tensorflow
scikit-learn
tqdm
gc
cv2
Pillow
Install dependencies using:

pip install -r requirements.txt
Usage
Steps to Run
Ensure the dataset azure.csv is in the project directory.
Run the Python script to preprocess data, train the models, and generate predictions:

python time_series_forecasting.py
Outputs
Plots for actual vs. predicted values for all metrics.
Printed performance metrics (RMSE, MAE, MAPE).
Project Highlights
Containerization: The models can be containerized using tools like Docker.
Scalability: Code is modular and can handle larger datasets by modifying TIME_STEPS and batch sizes.
Future Improvements
Add hyperparameter tuning to optimize the models further.
Experiment with additional recurrent architectures such as Bidirectional LSTMs or Transformers.
Integrate the pipeline into a cloud-based deployment system for real-time forecasting.
Author
Prasoona Ganaparthi
LinkedIn: linkedin.com/in/prasoona14
GitHub: github.com/prasoona14
