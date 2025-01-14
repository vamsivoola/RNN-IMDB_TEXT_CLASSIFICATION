# RNN-IMDB_TEXT_CLASSIFICATION

## Overview
RNN-IMDB_TEXT_CLASSIFICATION is a project that uses a Recurrent Neural Network (RNN) to classify IMDB movie reviews as **positive** or **negative**. It includes a user-friendly interface built with Streamlit for real-time sentiment analysis.

---

## Features
- **Text Preprocessing**: Tokenizes, pads, and decodes reviews for processing and interpretability.
- **Model Architecture**: A Simple RNN with embedding and dropout layers for sentiment classification.
- **Interactive App**: Provides a web-based interface for users to input reviews and receive predictions.

---

## Getting Started

### Installation

#### Prerequisites
1. Python 3.8 or higher.
2. Required Libraries:
   - `tensorflow`
   - `numpy`
   - `streamlit`
   - `re`

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RNN-IMDB_TEXT_CLASSIFICATION.git
   cd RNN-IMDB_TEXT_CLASSIFICATION
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

3. Ensure the pre-trained RNN model file (rnn_imdb.h5) is in the root directory.
4. Run the application:
     ```bash
     streamlit run app.py

## Model Architecture

| Layer         | Description                                    |
|---------------|------------------------------------------------|
| **Embedding** | Maps word indices to dense 128-dimensional vectors. |
| **RNN**       | Simple RNN with 128 neurons and ReLU activation. |
| **Dropout**   | Regularization layer with a 20% dropout rate.  |
| **Dense**     | Output layer with sigmoid activation for binary classification. |

---

## Example Output

### Input
**"Interstellar is one of the best movies that I have watched in the past decade."**

### Output
- **Sentiment**: Positive  
- **Prediction Score**: `0.77`

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Acknowledgements
- **IMDB Dataset**: [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **TensorFlow/Keras**: Framework for deep learning.
- **Streamlit**: Simplified app deployment.
