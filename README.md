## German NER example using Transformers. 
NER classifier for 3 classes on a custom German dataset with ~150K samples. Using huggingface pre-trained transformers and simpletransformers library. Training and final results  visualization with W&B.

## Jupyter notebook files
  File name                       | Description
-------------                   | -------------
`Data Parsing.ipynb`            | Parsing, fixing and preparing the data to train models.
`Training the model.ipynb`      | Training of different transormer architectures. 
`Summary of all models.ipynb`   | Shows the summary of all trained models (loss, f1 score, etc). Also runs and evaluates the best trained model.

## How to run
* Change the `path_working_dir` variable at the beginning of each notebook to your own working directory.
* This repository contains only the best trained transformer model `pytorch_model.bin` in `/models/deepset-gbert-base-epochs3`due to Github storage constrains.

## Weights & biases 
[Dashboard of all experiments](https://wandb.ai/roboserg/german-ner?workspace=user-roboserg)

[First overview report](https://wandb.ai/roboserg/german-ner/reports/First-initial-report--VmlldzozNjY3MTg)

## Best model general information
  Property                      | Description
-------------                   | -------------
model type | Transformer
model name used for finetuning             | deepset/gbert-base
model directory             | ../deepset-gbert-base-epochs3
model info | https://huggingface.co/deepset/gbert-base
paper | https://arxiv.org/pdf/2010.10906.pdf
release date | Oct 2020

## Best model training information
  Parameter                    | Value
-------------                | -------------
number of training epochs      | 3
training time | ~40 min on google colab
learning_rate | 1e-4
batch_size | 8

## Results of the best model:
  Metric                    | Value
-------------                | -------------
f1_score | 0.8666327741060837
precision | 0.8322213181448332
recall | 0.9040127275941312
eval_loss | 0.17615252890027014
training_loss | 0.14476392756443882
