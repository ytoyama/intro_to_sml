#!/bin/sh

. ~/anaconda3/bin/activate &&
jupyter nbconvert \
  --to notebook \
  --ExecutePreprocessor.timeout=-1 \
  --execute PS4-generative-models.ipynb \
  --inplace
