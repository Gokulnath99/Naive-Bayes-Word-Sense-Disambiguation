# Naive Bayes Word Sense Disambiguation

### Description

This repository features a Python implementation of the Naive Bayes algorithm for word sense disambiguation (WSD), focused on differentiating senses of polysemous words in textual contexts. The primary dataset revolves around the noun "plant" from the British National Corpus, with each instance annotated for specific senses.

### Key Features

- **Custom Naive Bayes Implementation:** Developed without external libraries like pandas, scikit-learn, or NLTK.
- **Cross-Validation:** Incorporates a five-fold cross-validation mechanism for robust testing and evaluation.
- **Add-One Smoothing:** Addresses zero-count scenarios in feature probability estimation.
- **Log Space Computation:** Utilizes log space for probability calculations to prevent underflow issues.
- **Extended Testing:** Additional evaluation using datasets for words "bass," "crane," "motion," "palm," and "tank."

### Usage

Run the program with the following command:
```sh
python WSD.py [dataset-file]
```
Outputs include fold-wise and average accuracies, and a detailed report (`[word].wsd.out`) of predicted senses for each test instance.

### Dataset

The primary dataset (`plant.wsd`) is sourced from the British National Corpus, providing rich, real-world examples for the word "plant" in various contexts. Additional datasets cover other polysemous words for comprehensive testing.

### Output

The script outputs accuracies for each fold and an average accuracy. A separate output file for each word lists the test instances with their predicted sense IDs.

### Components

- **WSD.py:** The main script for WSD using Naive Bayes.
