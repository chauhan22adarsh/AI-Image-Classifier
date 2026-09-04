# AI vs Real Image Classifier

A CNN-based classifier that predicts whether an image is AI-generated or a real
photograph, with a Streamlit app for trying it out on your own images.

## What's in here

- `Classifier.ipynb` - trains and evaluates the model
- `deploy.py` - Streamlit app that loads the trained model and classifies uploaded images
- `Model.h5` - the trained model (not committed here if it's too large; see below)

## Dataset

Trained on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
120k 32x32 images split evenly between real photos and AI-generated ones
(100k train, 20k test). Images this small keep training fast, but it's worth
being upfront that it also limits how much detail the model actually has to
work with - more on that under Limitations.

## Approach

A baseline CNN with light regularization overfit visibly after a few
epochs - training accuracy kept climbing while validation performance
flattened and then got worse. The fix: more dropout, L2 weight
regularization, and early stopping, plus light data augmentation (flips,
small rotations/zoom). L2/dropout values came from a small grid search
rather than a guess.

To isolate augmentation's actual effect, I also ran the same regularized
architecture with augmentation turned off, holding everything else fixed -
and as a fourth comparison point, fine-tuned a frozen pretrained ResNet50
(ImageNet weights, small trainable head) to see how transfer learning
stacks up against training from scratch here.

All four models are evaluated on accuracy, precision/recall/F1, confusion
matrix, and ROC/AUC, plus threshold tuning away from the default 0.5 cutoff.
I also went through misclassified test images by hand, used Grad-CAM to
see what the model was actually focusing on for those, and ran a small
out-of-distribution test on images completely outside CIFAKE to get a real
number on how well this generalizes past the benchmark.

## Results

CIFAKE test set (20k images):

| Model | Test accuracy | AUC |
|---|---|---|
| Baseline CNN (overfit) | 86.63% | 0.968 |
| Regularized CNN + augmentation | 90.93% | 0.976 |
| **Regularized CNN, no augmentation (final model)** | **93.28%** | **0.983** |
| ResNet50 (frozen, transfer learning) | 76.50% | 0.842 |

Out-of-distribution (own images, outside CIFAKE): 2/2 = 100%, small sample.

The regularized model without augmentation wins clearly and is what's saved
as `Model.h5`. It beats the overfit baseline while showing none of the
baseline's overfitting spike, which suggests augmentation - not the
dropout/L2/early stopping - was the actual drag on accuracy in the augmented
run: flips/rotations/zoom on 32x32 images likely wash out low-level pixel
artifacts (compression patterns, generator noise) that separate FAKE from
REAL. ResNet50 underperforms both from-scratch CNNs, which tracks -
its frozen ImageNet features are tuned for object recognition, not the
statistical fingerprints that distinguish real photos from generated ones.

(Re-ran this and got slightly different numbers each time - GPU ops aren't
perfectly reproducible even with a fixed seed - but the ranking between
models and the winning model's AUC held steady across runs.)

## Running it Locally

```
pip install -r requirements.txt
streamlit run deploy.py
```

Needs `Model.h5` sitting in the same folder - it's loaded at startup.

## Retraining

`Classifier.ipynb` expects CIFAKE's train/ and test/ folders (each with
FAKE/ and REAL/ subfolders). Paths default to a Kaggle-style layout
(`/kaggle/input/cifake-real-and-ai-generated-synthetic-images/`) but run
fine locally too - just swap those two path strings. I ran mine on a Mac;
TensorFlow picks up Metal/GPU automatically if available, otherwise falls
back to CPU (slower, still fine at 32x32).

Gotcha on newer Keras (hit this on 3.15): reading `.inputs`/`.output`
directly off a `Sequential` model in the Grad-CAM section can throw
`AttributeError: has never been called`, since `Sequential` doesn't track
its call graph the way a functional model does. Fixed by rebuilding the
forward pass through a fresh `Input` using the same trained layers instead.

## Limitations

- 32x32 input is small enough that a lot of the finer artifacts that
  distinguish AI images from real ones probably aren't visible to the model
  at all. Accuracy on this benchmark doesn't necessarily carry over well to,
  say, a high-resolution modern diffusion image uploaded through the app -
  worth keeping in mind if you actually try it on random images off the
  internet. The out-of-distribution section in the notebook puts an actual
  number on this rather than leaving it as a guess.
- The app resizes whatever you upload down to 32x32 before predicting, so
  there's a real gap between what the model was trained/evaluated on and
  what it sees in practice.
- Only trained on CIFAKE, so newer generators not represented in that
  dataset may not be classified reliably.
- The ResNet50 comparison keeps the pretrained layers frozen and only trains
  a small head on top - I didn't attempt fine-tuning the later ResNet
  layers, which usually helps but needs more compute and more care around
  overfitting than I had time budget for here.
