# DLNLP_assignment_25
This project focuses on two natural language processing (NLP) classification tasks using transformer-based models (BERT and RoBERTa):

Task A: Sentiment polarity classification on the Sentiment140 dataset (binary: positive vs. negative)

Task B: Emotion classification on a 7-class GoEmotions subset (e.g., anger, joy, sadness)

.
├── A.py                   # Script for Task A: Sentiment Classification (BERT)
├── B.py                   # Script for Task B: Emotion Classification (RoBERTa)
├── main.py                # Optional main launcher (if combining)
├── datasets/              # Folder containing all raw and processed datasets
│   ├── sentiment140_sampled_10000_each.csv    # Task A Train Dataset
│   └── sentiment140_test_2000.csv             # Task A Test Dataset
│   └── train.tsv                              # Task B Dataset
