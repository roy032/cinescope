Hand-labelled test shots: 289 from the held-out videos ({'ECS': 0, 'CS': 27, 'MS': 204, 'FS': 34, 'LS': 24}); majority-class accuracy 0.706.

| System | Accuracy [95% CI] | Mean class acc. | Shots scored |
|---|---|---|---|
| SigLIP zero-shot (prompts only) | 0.792 [0.74–0.84] | 0.524 | 289 (100%) |
| Face size only, shots with a face | 0.668 [0.61–0.73] | 0.643 | 190 (66%) |
| Teacher: prompts + face size | 0.806 [0.76–0.85] | 0.555 | 289 (100%) |
| ResNet-18, ImageNet init | 0.796 [0.75–0.84] | 0.540 | 289 (100%) |
| ResNet-18, random init | 0.775 [0.73–0.82] | 0.523 | 289 (100%) |
