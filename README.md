# PDRF

Official implementation of **PDRF**.

## Training

To train PDRF on the Shanghai Radar dataset, run:

```bash
python train.py --dataset shanghai --frames_in 5 --frames_out 20 \
    --batch_size 8 --epochs 300 --valid --wandb_state disabled
```

## Inference

To run inference with a pretrained checkpoint, run:

```bash
python inference.py --dataset shanghai \
    --ckpt_milestone /path/to/ckpt-best-shanghai.pt
```

Replace `/path/to/ckpt-best-shanghai.pt` with the path to your downloaded checkpoint.

## Datasets

All four datasets used in our paper follow the data preparation settings of [DiffCast](https://github.com/DeminYu98/DiffCast).

The datasets are available from the following sources:

- [SEVIR](https://nbviewer.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb)
- [MeteoNet](https://meteofrance.github.io/meteonet/english/data/rain-radar/)
- [Shanghai Radar](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2GKMQJ)
- [CIKM Radar](https://tianchi.aliyun.com/dataset/1085)

## Pretrained Checkpoints

Pretrained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1xxUQxXm4sdp-FR9c6RwqU43YHe8b-Rny?hl=zh-cn).

## Acknowledgements

We thank the authors of [DiffCast](https://github.com/DeminYu98/DiffCast) for making their code and data-processing resources publicly available.
