# Improvise a Jazz Solo with an LSTM Network

## Project Overview

This project focuses on generating jazz music using deep learning, specifically an LSTM-based (Long Short-Term Memory) neural network. The goal is to create a system capable of improvising jazz solos in a style similar to a trained dataset.

### Objectives:
- Apply an LSTM to a music generation task.
- Generate original jazz music based on learned patterns.
- Utilize the Functional API of TensorFlow/Keras to design and train a complex sequential model.

## Dataset

The model is trained on a corpus of jazz music preprocessed into sequences of musical "values." These values represent musical notes or chords encoded as one-hot vectors. The dataset is formatted as:
- `X`: A (m, T_x, 90) array where each sequence contains 30 notes (T_x = 30) with 90 possible values.
- `Y`: The labels shifted by one time step to predict the next value.

## Methodology

1. **Preprocessing**:
   - Musical pieces are converted into sequences of values.
   - Data is split into inputs (`X`) and corresponding outputs (`Y`).

2. **Model Architecture**:
   - Uses an LSTM network with a hidden state size of 64.
   - Inputs are reshaped to match the LSTM requirements.
   - A dense layer with softmax activation is used for output predictions.

3. **Training**:
   - Trains on snippets of 30 musical values.
   - Optimizes to predict the next note in a sequence using categorical cross-entropy loss.

4. **Generation**:
   - Implements a custom sequence generation loop.
   - At each step, the predicted note is fed as input to generate the next note iteratively.

### Tools and Libraries
- TensorFlow/Keras
- `music21` for handling MIDI data.
- Supporting modules like `preprocess.py` and `music_utils.py` for data preparation.

## Results
After training, the model generates jazz solos by predicting notes iteratively. The output can be converted back into MIDI format for playback.

## Usage

### Prerequisites:
- Python 3.x
- Required libraries: TensorFlow, NumPy, `music21`, Matplotlib

### Steps to Run:
1. Clone the repository or download the project files.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb
   ```
4. Train the model on the provided dataset.
5. Use the trained model to generate jazz solos.

### Outputs:
- Generated MIDI files can be played to listen to the jazz solo.

## Future Work
- Extend the model to handle other genres of music.
- Experiment with more advanced architectures like transformers for better results.
- Incorporate real-time generation and visualization of music.


