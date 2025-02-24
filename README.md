
## Get Started

### Data preparation
Like [BasicSR](https://github.com/XPixelGroup/BasicSR), we put the training and validation data under the datasets folder. You can download the data in the way provided by [BasicSR](https://github.com/XPixelGroup/BasicSR), but please note that our data path is slightly different from BasicSR, please modify the data path in the configuration file (all configuration files are under the [configs](configs) folder).

Also, under the [scripts/datasets/DIV2K](scripts/datasets/DIV2K) folder, we provide the script to download the DIV2K dataset. You can download the DIV2K dataset as follows:

```bash
bash scripts/datasets/DIV2K/build.sh
```

Finally, the structure of the dataset is as follows:

```
datasets/
├── DIV2K
    ├── train
        ├── HR
            ├── original
                ├── 0001.png
                ├── 0002.png
                ├── ...
                ├── 0800.png
            ├── subs
                ├── 0001_s001.png
                ├── 0001_s002.png
                ├── ...
                ├── 0800_s040.png
        ├── LR
            ├── bicubic
                ├── X2
                    ├── original
                        ├── 0001x2.png
                        ├── 0002x2.png
                        ├── ...
                        ├── 0800x2.png
                    ├── subs
                        ├── 0001_s001.png
                        ├── 0001_s002.png
                        ├── ...
                        ├── 0800_s040.png
                ├── X3...
                ├── X4...
    ├── valid...
├── Set5
    ├── GTmode12
        ├── baby.png
        ├── bird.png
        ├── butterfly.png
        ├── head.png
        ├── woman.png
    ├── original...
    ├── LRbicx2...
    ├── LRbicx3...
    ├── LRbicx4...
```

### Training

[train.py](train.py) is the entry file for the training phase. You can find the description of [train.py](train.py) in the [BasicSR](https://github.com/XPixelGroup/BasicSR) repository. The training command is as follows:
```bash
python train.py -opt {configs-path}.yml
```
where `{configs-path}` represents the path to the configuration file. All configuration files are under the [configs/train](configs/train) folder. The `log`, `checkpoint` and other files generated during training are saved in the [experiments](./experiments)`/{name}` folder, where `{name}` refers to the name option in the configuration file.


### Convert
ETDS during training is a dual stream network, and it can be converted into a plain model through [converter.py](converter.py), as follows:
```bash
python converter.py --input {input-model-path}.pth --output {output-model-path}.pth
```
where `{input-model-path}.pth` represents the path to the pre-trained model, and `{output-model-path}.pth` indicates where the converted model will be saved.

Also, the code of converting ECBSR and ABPN to plain models is in [converter_ecbsr_et](converter_ecbsr_et.py) and [converter_abpn_et](converter_abpn_et.py).

Our pretrained models after conversion are in the [experiments/pretrained_models](./experiments/pretrained_models) folder.

### Validition
The validition command is as follows:
```bash
python test.py -opt {configs-path}.yml
```
where `{configs-path}` represents the path to the configuration file. All configuration files are under the [configs/test](configs/test) folder. The verification results are saved in the [results](./results) folder.

### Results

<details>
<summary>Mobile Image Super-Resolution</summary>
<p align="center">
  <img width="900" src="./asserts/tables/table-all.png">
</p>
</details>

<details>
<summary>ETDS, ECBSR and ABPN with and without ET</summary>
<p align="center">
  <img width="900" src="./asserts/tables/table-ablation-1.png">
  <img width="900" src="./asserts/tables/table-ablation-2.png">
</p>
</details>
<!-- 
<details>
<summary>others (e.g., ECBSR and ABPN) with and without ET</summary>
<p align="center">
</p>
</details> -->

## ✨Core File List
This repository is based on [BasicSR](https://github.com/XPixelGroup/BasicSR)'s code framework and has undergone secondary development. Here we point out the core files of this repository (Descending order of importance):

- [core/archs/ir/ETDS/arch.py](core/archs/ir/ETDS/arch.py) : ETDS architecture
- [converter.py](converter.py) : Convert ETDS to plain model using the Equivalent Transformation technique.
- [converter_abpn_et.py](converter_abpn_et.py) : Convert ABPN to plain model using the Equivalent Transformation technique.
- [converter_ecbsr_et.py](converter_ecbsr_et.py) : Convert the ECBSR to a model with less latency using the Equivalent Transformation technique.
- [train.py](train.py) : Model training pipeline
- [test.py](test.py) : Model testing pipeline
- [core/models/ir/etds_model.py](core/models/ir/etds_model.py) : An implementation of the model interface for pipeline calls.
- [validation/reparameterization.py](scripts/validation/reparameterization.py)😊: Verify the correctness of the description of reparameterization in the appendix.
