# AI vs Real Image Classifier

A CNN-based classifier that predicts whether an image is AI-generated or a real
photograph, with a Streamlit app for trying it out on your own images.

## What's in here

- `Classifier.ipynb` - trains and evaluates the model
- `deploy.py` - Streamlit app that loads the trained model and classifies uploaded images
- `Model.h5` - the trained model (not committed here if it's too large; see below)
- `.devcontainer/` - config so this runs out of the box in a GitHub Codespace

## Dataset

Trained on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
120k 32x32 images split evenly between real photos and AI-generated ones
(100k train, 20k test). Images this small keep training fast, but it's worth
being upfront that it also limits how much detail the model actually has to
work with - more on that under Limitations.

## Approach

I trained a baseline CNN first with light regularization, and it overfit
pretty visibly after a handful of epochs - training accuracy kept climbing
while validation performance flattened out and then got worse. That's the
usual sign the model was starting to memorize the training set instead of
learning anything that generalizes.

From there I built a second version of the same architecture with more
dropout, L2 weight regularization, and early stopping tied to validation
loss, plus some light data augmentation (flips, small rotations/zoom) on the
training data. Curves for both versions are in the notebook so the
before/after is visible rather than just asserted.

There's also a quick ablation in the notebook where I retrained the
regularized architecture with augmentation turned off, mainly to check
whether augmentation or the regularization itself was driving any change in
test accuracy between versions. Wanted an actual answer to that instead of
guessing.

Evaluation covers accuracy, precision/recall/F1, a confusion matrix, and
ROC/AUC. I also went through a batch of misclassified test images by hand -
some interesting patterns there, described more in the notebook, but broadly
a chunk of the errors seem to come from images that are just hard to tell
apart even for a person at 32x32 resolution.

## Results

*(Fill in after running the notebook - see the note below on where each
number comes from. Don't guess at these, pull them straight from the
notebook's output cells.)*

| Model | Test accuracy | F1 | AUC |
|---|---|---|---|
| Baseline CNN (overfit) | 0.8629 | 0.86 | 0.9739 |
| Improved CNN (regularized + augmented) | 0.8822 | 0.88 | 0.9713 |
| Improved CNN (regularized, no augmentation) | 0.9419 | 0.94 | 0.9862 |

## Running it

### Option 1: GitHub Codespaces

Push this repo to GitHub, click Code > Codespaces > Create codespace. The
devcontainer installs everything and starts the Streamlit app automatically
on port 8501.

### Option 2: Locally

```
pip install -r requirements.txt
streamlit run deploy.py
```

Needs `Model.h5` sitting in the same folder - it's loaded at startup.

## Retraining

`Classifier.ipynb` expects to run on Kaggle with the CIFAKE dataset added as
input (path is hardcoded to `/kaggle/input/cifake-real-and-ai-generated-synthetic-images/`).
If you want to run it somewhere else, just point those two path strings
(train/test) at wherever you've put the data locally.

## Limitations

- 32x32 input is small enough that a lot of the finer artifacts that
  distinguish AI images from real ones probably aren't visible to the model
  at all. Accuracy on this benchmark doesn't necessarily carry over well to,
  say, a high-resolution modern diffusion image uploaded through the app -
  worth keeping in mind if you actually try it on random images off the
  internet.
- The app resizes whatever you upload down to 32x32 before predicting, so
  there's a real gap between what the model was trained/evaluated on and
  what it sees in practice.
- Only trained on CIFAKE, so newer generators not represented in that
  dataset may not be classified reliably.
