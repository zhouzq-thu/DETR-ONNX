DETR models in ONNX format
==========================

Currently supports:
- [x] [DETR](https://github.com/facebookresearch/detr), 2020
- [x] [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), 2021
- [x] [LW-DETR](https://github.com/Atten4Vis/LW-DETR), 2024
- [x] [RT-DETR](https://github.com/lyuwenyu/RT-DETR), 2024
- [x] [RF-DETR](https://github.com/roboflow/rf-detr), 2025

> **NOTE**: The normalization using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] is fused in models.

## Model info

- Inputs:
  - input: ['N', 3, 'H', 'W'] (FLOAT)
- Outputs:
  - logits: ['N', 300, 80 or 91] (FLOAT)
  - boxes: ['N', 300, 4] (FLOAT)

You can use the following script to get model infos.

```python
import onnxruntime as ort

sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
output_names = [output.name for output in sess.get_outputs()]
input_shape = sess.get_inputs()[0].shape
meta = sess.get_modelmeta().custom_metadata_map
stride = int(meta.get('stride', -1))
class_names = eval(meta.get('names', '{}'))
```

## Usage

```python
import cv2 as cv
from detr_onnx import DetrONNX

model_path = "models/rt-detrv2.onnx"
detr = DetrONNX(model_path)

img = cv.imread("images/kite.jpg")
prob, boxes = detr.detect(img)
res = detr.plot_result(img, prob, boxes)

cv.imshow("res", res)
cv.waitKey(0)
```
