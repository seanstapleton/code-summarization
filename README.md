# Code Summarization

## Preprocessing Data
Data for this model must be preprocessed into lists of paths from an AST tree. We have written a suite of preprocessing tools to do this for you. To preprocess your data, begin by formatting it into .txt files of training examples separated by new lines and parallel labels separated by new lines. Now you can run the following command:

```python preprocess.py <path_to_dataset_txt> <path_to_labels_txt> <path_to_output_directory>```