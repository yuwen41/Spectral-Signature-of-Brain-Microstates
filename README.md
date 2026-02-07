# Spectral Signature of Brain Microstates

## Overview

This project analyzes the **spectral signatures of brain microstates** across different psychological and physiological conditions, specifically comparing **Anxiety** and **Chronic Stress** states. Through deep learning approaches, we investigate which model architecture best captures the spatial-spectral characteristics of EEG microstates.

## Key Findings

📊 **CNN outperforms LSTM** for spectral signature classification

- **CNN (Convolutional Neural Networks)** achieves superior performance, indicating that the **spatial-frequency features at each time point** of the spectrogram are more informative than temporal dependencies
- **LSTM (Long Short-Term Memory)** shows lower performance, suggesting that the temporal sequence information in spectrograms is less critical for this task
- **Implication**: The distinctive patterns in brain microstates are encoded primarily in the **frequency domain features** rather than temporal dynamics

## Project Structure

```
Spectral-Signature-of-Brain-Microstates/
├── Anxiety/
│   └── [Analysis and preprocessing scripts for anxiety state data]
├── Chronic Stress/
│   └── [Analysis and preprocessing scripts for chronic stress state data]
└── README.md
```

### Directory Descriptions

- **Anxiety/**: Contains data processing and analysis code for anxiety condition EEG records
- **Chronic Stress/**: Contains data processing and analysis code for chronic stress condition EEG records

## Methodology

### Data Processing
- Extract EEG spectrograms from raw signals
- Compute spectral signatures of brain microstates
- Prepare normalized datasets for model training

### Model Comparison
- **CNN Architecture**: Captures spatial-spectral patterns in spectrograms
- **LSTM Architecture**: Processes temporal sequences of spectrogram features
- **Evaluation**: Compare classification accuracy, precision, recall, and other relevant metrics

## Getting Started

### Requirements
- Python 3.7+
- TensorFlow/Keras
- NumPy
- SciPy
- Matplotlib
- scikit-learn

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. **Run analysis for Anxiety condition:**
   ```bash
   python Anxiety/[script_name].py
   ```

2. **Run analysis for Chronic Stress condition:**
   ```bash
   python Chronic\ Stress/[script_name].py
   ```

3. **Train and evaluate models:**
   - Update script paths and dataset locations in the scripts
   - Execute to generate results and visualizations

## Results

The analysis demonstrates that:
- **Spatial-frequency features** in spectrograms are the primary discriminators between conditions
- CNN's convolutional layers effectively extract local spectral patterns
- LSTM's sequential processing does not provide additional benefit for this task
- This suggests brain microstates have **stable spectral signatures** that don't require temporal context for classification

## Visualizations

Expected outputs include:
- Spectrograms for anxiety vs. chronic stress conditions
- Model performance comparison plots
- Feature importance/activation maps (for CNN)
- Classification metrics and confusion matrices

## Future Work

- [ ] Explore hybrid architectures combining spatial and temporal features
- [ ] Investigate attention mechanisms for feature weighting
- [ ] Expand to additional psychological/neurological conditions
- [ ] Perform interpretability analysis on learned CNN filters
- [ ] Validate findings with larger datasets

## References

- Key research on brain microstates and EEG analysis
- Deep learning applications in neuroimaging
- CNN vs. RNN for time-series classification in medical domains

## License

[Specify your license here, e.g., MIT, GPL, etc.]

## Author

Yu Wen (yuwen41)

## Contact

For questions or collaborations, please open an issue or contact the repository owner.