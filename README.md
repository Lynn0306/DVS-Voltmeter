# DVS-Voltmeter

Code repo for the paper 'DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors'.

## Prerequisites
```
easydict == 1.9
pytorch >= 1.8
numpy >= 1.20.1
opencv-python == 4.5.1.48
tqdm == 4.49.0
```
The code may be compatible with lower versions, while the aforementioned ones have been tested.

## Get Started

Put videos in ***data_samples/original*** folder and modify the data index tree as following:

```
├── [data_samples]
│   ├── original
│   │   ├── videoname1
│   │   │   ├── info.txt
│   │   │   ├── framename11.png
│   │   │   ├── framename12.png
│   │   ├── videoname2
│   │   │   ├── info.txt
│   │   │   ├── framename21.png
│   │   │   ├── framename22.png
```

The video info file ***info.txt*** records the path and timestamp ($\mu s$) of each frame.

## Usage

1. Configure the src/config.py file. For detailed configuration, please refer to the config file.
2. Run ```python main.py```

## Biblography

If you find our work useful, please use the following citation.
```
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
```

## License

MIT License