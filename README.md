# Indian Lentils Image Classifier

A 20-class image classifier for Indian lentils/legumes, built as a vehicle for a
correctly-evaluated transfer-learning baseline and a GPU performance
investigation. See [`Analysis.md`](Analysis.md) for the full write-up.

## Contents

| File | Description |
|---|---|
| [`Lentil_Classifier_NN.ipynb`](Lentil_Classifier_NN.ipynb) | Training notebook: data loading/splitting, model training (EfficientNetB0 baseline, ResNet18 comparison), and GPU performance benchmarking (mixed precision, `torch.compile`). |
| [`Analysis.md`](Analysis.md) | Full write-up of the methodology, results, and GPU performance findings. |
| [`model-card.md`](model-card.md) | Model card for the trained classifier. |
| `resnet_lentils.pth` | Trained model weights. Not tracked in this repo (see `.gitignore`) — published instead to the Hugging Face Model Hub, linked below. |

## Deployed model & demo

- **Model weights:** [silversur4/indian-lentils-image-classifier](https://huggingface.co/silversur4/indian-lentils-image-classifier) — Hugging Face Model Hub.
- **Live demo:** [Instant-Gram](https://huggingface.co/spaces/silversur4/Instant-Gram) — Gradio Space; upload a photo or use your camera to classify a lentil/bean/chickpea in real time.

The Space downloads weights from the Model Hub repo at startup rather than
bundling them directly, keeping the deployed app small. The deploy code for the Space lives in a separate repository.
