# DDIM

The Python code is used to inference a pretrained diffusion model for human face generation using DDIM.

## Usage

### Inferencing
Pretrained model:
```bash
bash get_dataset.sh
bash hw2_download_ckpt.sh
```

Inference:
```bash
bash hw2_1.sh <noise_folder> <pretrained_unet_path> <save_path>
```

Interpolation:
```bash
python3 slerp.py
```
In `slerp.py` changing the `function=0/1` can change different noise interpolation mode (slerp/linear) 

## Reference
[DDIM DLCV](https://docs.google.com/presentation/d/1nWH_CmF6iba0kQmi0TV_yI2Emu1EvjrX/edit#slide=id.p7)

