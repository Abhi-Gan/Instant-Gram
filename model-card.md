---
license: mit
pipeline_tag: image-classification
library_name: pytorch
tags:
  - image-classification
  - resnet
  - transfer-learning
  - pytorch
  - agriculture
model-index:
  - name: indian-lentils-image-classifier
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Indian Lentils Dataset
          type: indian-lentils
        metrics:
          - type: accuracy
            value: 0.999
            name: Held-out eval accuracy
---

# Indian Lentils Image Classifier

ResNet18, fine-tuned to classify 20 varieties of Indian lentils, beans, and
chickpeas from a photo.

## Model details

- **Base model:** `resnet18`, ImageNet-pretrained (torchvision `ResNet18_Weights.DEFAULT`), fine-tuned end-to-end.
- **Input:** 224×224 RGB image, ImageNet mean/std normalization (`Resize(256)` → `CenterCrop(224)`).
- **Output:** 20-way classification over the classes listed below.
- **Task:** image-classification

## Training data

[Indian Lentils dataset](https://data.mendeley.com/datasets/r4yhfnzc5y/1) (Mendeley Data) —
10,000 images across 20 classes, split 80/20 (train/eval), stratified by class.

Classes: Black_Beans, Black_EyedPea, Brown_Chickpea, Desi_chickpea, Horse_Gram,
Husked_BlackGram, Husked_RedLentil, Kidney_Bean, Red_Beans, Red_KidneyBean,
Red_chickpea, Soya_Bean, White_Chickpea, White_beans, White_peabeans,
Whole_BlackGram, Whole_GreenGram, Whole_TurkishGram, chana1, chana2.

## Evaluation results

| Metric | Value |
|---|---|
| Accuracy (held-out, 2,000 images) | 99.9% |
| Macro F1 | 1.00 |

Residual misclassifications were concentrated among visually similar
look-alike classes (e.g. `Kidney_Bean` / `Red_Beans` / `Red_KidneyBean`),
consistent with genuine generalization rather than a data-leakage artifact.

A reference benchmark study over 18 CNN architectures on this same dataset
concluded EfficientNetB0 was the strongest performer; this model matches that
accuracy while training ~1.55x faster per epoch on an NVIDIA T4, via
per-architecture learning-rate tuning rather than a new technique. Full
methodology and GPU performance analysis (mixed precision, `torch.compile`,
cross-architecture comparison): see the project write-up[^1].

[^1]: link to your GitHub repo / README here

## How to use

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image

weights_path = hf_hub_download(
    repo_id="silversur4/indian-lentils-image-classifier",
    filename="resnet_lentils.pth",
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 20)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("your_photo.jpg").convert("RGB")
x = transform(image).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
```

## License

MIT.
