# Wrestling Highlight Detector - Groot Pipeline

A machine learning project to automatically detect highlights in wrestling matches using a sophisticated 4-stage deep learning pipeline.

## Project Evolution

### Phase 1: Live Filter (Legacy - Not Used)
- Basic live vs non-live detection using manual labeling
- Terminal-based range labeling tools
- Database management system
- **Note**: This implementation is preserved in `live_filter/` directory but is not part of the current Groot pipeline

### Phase 2: Groot LSTM Pipeline (Current Focus)
Advanced LSTM-based highlight detection with modular, testable components.

## Groot Pipeline Architecture

The Groot pipeline is divided into modular, independently testable components:

1. **Data Processing** - Video preprocessing and dataset preparation
2. **Segmentation Model** (DeepLabV3+) - Extract wrestler silhouettes
3. **2D CNN Feature Extractor** (EfficientNet-B3) - Extract per-frame features
4. **Bi-LSTM Temporal Modeling** - Capture temporal patterns
5. **Dense Classifier** - Final highlight/non-highlight classification
6. **Connector Training Script** - Orchestrates all 4 model architectures
7. **Training Phase** - Full pipeline training with cross-validation
8. **Testing Phase** - Evaluation and deployment

## Development Philosophy

### Modular Development Approach
- **Independent Testing**: Each component must work standalone before integration
- **Incremental Progress**: No jumping to new parts until previous ones are guaranteed to work
- **Sub-pipeline Validation**: Complex sub-components are tested at smaller scale first
- **Elegant Problem Solving**: Address issues while working on smaller, modular components
- **Error Prevention**: Avoid massive debugging sessions by catching issues early

### Development Habits
- Test each component thoroughly before moving to the next
- Create comprehensive test cases for each module
- Document any deviations or sub-pipeline complexities
- Maintain clear separation of concerns between components
- Use configuration files for easy parameter tuning
- Implement proper logging and error handling at each stage

## Project Structure

```
wrestling-highlight-detector/
├── groot/                    # Groot LSTM Pipeline (Current Focus)
│   ├── config.py              # All hyperparameters and settings
│   ├── dataset_loader.py      # Dataset download and setup
│   ├── prepare_training_data.py  # Data preparation pipeline
│   ├── models/
│   │   ├── segmentation.py    # DeepLabV3+ implementation
│   │   ├── cnn_feature_extractor.py  # EfficientNet implementation
│   │   ├── lstm_temporal.py   # Bi-LSTM implementation
│   │   └── classifier.py      # Dense classification network
│   ├── data/
│   │   ├── dataset.py         # PyTorch dataset classes
│   │   └── transforms.py      # Data augmentation
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualization.py   # Plotting utilities
│   ├── inference/
│   │   └── predictor.py       # Inference pipeline
│   └── utils/
│       ├── database.py        # Database operations
│       └── video_utils.py     # Video processing utilities
├── live_filter/               # Phase 1: Live vs Non-Live Detection (Legacy - Not Used)
├── highlights/                # Phase 2: Highlight Detection (Legacy)
├── database/                  # Database management
├── archives/                  # Old/utility scripts
├── match_directory/           # Original match videos
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Current Status

**Active Development**: Data Processing Pipeline (Part 1 of Groot)

The project is currently focused on implementing the data processing component of the Groot pipeline in the `groot/` directory. This foundational component will handle:
- Video preprocessing and frame extraction
- Dataset preparation and validation
- Database integration for training data
- Sub-pipeline testing and validation

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Groot-specific dependencies:
```bash
pip install kaggle
pip install segmentation-models-pytorch
pip install torch torchvision
pip install tqdm matplotlib seaborn
```

4. Initialize the database:
```bash
python database/database_setup.py
```

## Next Steps

1. **Complete Data Processing Pipeline** - Implement and test video preprocessing
2. **Segmentation Model** - Develop and validate DeepLabV3+ implementation
3. **Feature Extractor** - Build EfficientNet-B3 feature extraction
4. **LSTM Temporal Modeling** - Implement Bi-LSTM for temporal patterns
5. **Dense Classifier** - Create final classification network
6. **Integration Testing** - Connect all components
7. **Training Phase** - Full pipeline training
8. **Testing Phase** - Evaluation and deployment

## Database Schema

### Legacy Tables (Phase 1 - Not Used in Groot)
- `original_matches`: Match video metadata
- `match_ranges`: Time ranges labeled as LIVE or NOT LIVE
- `labels`: Individual clip labels
- **Note**: These tables exist from the previous implementation but are not used in the Groot pipeline

### Groot Tables (Phase 2)
- `lstm_training_clips` - New labeled training data
- `lstm_models` - Model versions and checkpoints
- `lstm_training_runs` - Training session metadata
- `lstm_predictions` - Clip-level predictions
- `lstm_match_predictions` - Full match sliding window results
- `lstm_features` - Intermediate features for analysis

## Development Guidelines

- **Test First**: Always test components independently
- **Document Changes**: Update this README with any pipeline modifications
- **Modular Design**: Keep components loosely coupled
- **Error Handling**: Implement robust error handling at each stage
- **Configuration**: Use config files for easy parameter management
- **Logging**: Maintain comprehensive logs for debugging 