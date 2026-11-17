# Anomaly Segmentation for Road Scenes with Mask Architectures

Group project at Politecnico di Torino on **semantic segmentation** and **pixel-level anomaly (out-of-distribution) detection** for autonomous driving.
We compare a classic per-pixel network (**ERFNet**) with a modern mask-based transformer (**EoMT**, DINOv2 backbone), study how to adapt EoMT across domains (COCO → Cityscapes), and evaluate how well each model spots unexpected objects on the road.

> Fork of [FrancescaFox23/Project_fundamentals](https://github.com/FrancescaFox23/Project_fundamentals), itself built on the course starter code by [Alessandro Marinai](https://github.com/AlessandroMarinai/MaskArchitectureAnomaly_CourseProject).

---

## What we did

| Step | Task | Notebook |
|---|---|---|
| 4 | Cross-domain inference: EoMT trained on Cityscapes (semantic) vs EoMT trained on COCO (panoptic), evaluated on Cityscapes | [`eomt/inference.ipynb`](eomt/inference.ipynb) |
| 5 | Fine-tuning the COCO checkpoint on Cityscapes: head-only, partial backbone unfreezing, LoRA | [`eomt/inference.ipynb`](eomt/inference.ipynb) |
| 7 | Anomaly segmentation baseline with ERFNet (MSP, MaxLogit, Entropy) | [`eval/evalAnomalyStep7.ipynb`](eval/evalAnomalyStep7.ipynb) |
| 8 | Anomaly segmentation with EoMT (MSP, MaxLogit, MaxEntropy, RbA) on three checkpoints, plus temperature scaling | [`eval/evalAnomalyStep8.ipynb`](eval/evalAnomalyStep8.ipynb) |

## Key results

### Semantic segmentation (Cityscapes val, 500 images)

| Model | mIoU |
|---|---|
| EoMT – Cityscapes checkpoint | **81.7%** |
| EoMT – COCO panoptic checkpoint (classes mapped to Cityscapes) | 50.1% |
| COCO → Cityscapes fine-tuning, head only | 74.1% |
| COCO → Cityscapes fine-tuning, partial backbone unfreezing | 74.8% |
| COCO → Cityscapes fine-tuning, LoRA | **76.3%** |

Fine-tuning recovers most of the domain gap (+26 mIoU points over the raw COCO model), with LoRA giving the best result on a single consumer GPU.

### Anomaly segmentation (AuPRC ↑ / FPR95 ↓, %)

| Dataset | ERFNet (best: MaxLogit) | EoMT Cityscapes (MSP) |
|---|---|---|
| SMIYC RoadAnomaly21 | 38.3 / 59.4 | **68.6 / 14.4** |
| SMIYC RoadObstacle21 | 4.6 / 48.5 | **84.1 / 4.6** |
| Fishyscapes Lost & Found | 3.3 / 45.5 | **16.8 / 9.9** |
| Fishyscapes Static | 9.4 / **40.3** | **62.5** / 43.4 |

Main findings:
- The mask-based architecture is far better at detecting anomalies than the per-pixel baseline, with much higher AuPRC and lower false-positive rates on most benchmarks.
- For ERFNet, **MaxLogit** is the most reliable score on every dataset; for the in-domain EoMT checkpoint, plain **MSP** is the most consistent.
- The COCO checkpoint performs poorly on road anomalies; after fine-tuning on Cityscapes, **MaxEntropy** becomes the best score on the SMIYC benchmarks (75.0% AuPRC on RoadAnomaly21).
- Temperature scaling on MSP gives only small gains (e.g. 70.4% AuPRC on RoadAnomaly21 at T = 2.0).

Full per-method tables for all three EoMT checkpoints are in the Step 8 notebook.

## Repository structure

```
eomt/            EoMT code (training, configs, datasets) + inference & fine-tuning notebook
eval/            ERFNet evaluation tools and anomaly segmentation notebooks
trained_models/  Pretrained ERFNet weights used for the baseline
```

See the README inside each folder for setup instructions. Requirements for the main notebooks are in [`eomt/requirements.txt`](eomt/requirements.txt).

## Tech stack

Python · PyTorch · PyTorch Lightning · DINOv2 / EoMT · LoRA · Weights & Biases · Cityscapes, COCO, SMIYC, Fishyscapes, Road Anomaly
