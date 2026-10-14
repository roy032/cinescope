Hand-labelled test shots: 51 from the held-out videos ({'ECS': 7, 'CS': 13, 'MS': 14, 'FS': 5, 'LS': 12}); majority-class accuracy 0.275.

| System | Accuracy [95% CI] | Mean class acc. | Shots scored |
|---|---|---|---|
| open_clip ViT-B/32 zero-shot (prompts only) | 0.412 [0.27–0.55] | 0.379 | 51 (100%) |
| Face size only (OpenCV YuNet), shots with a face | 0.158 [0.00–0.32] | 0.155 | 19 (37%) |
| Teacher: prompts + face size | 0.431 [0.29–0.57] | 0.396 | 51 (100%) |
| ResNet-18, ImageNet init | 0.176 [0.08–0.29] | 0.131 | 51 (100%) |
| ResNet-18, random init | 0.137 [0.06–0.24] | 0.100 | 51 (100%) |
