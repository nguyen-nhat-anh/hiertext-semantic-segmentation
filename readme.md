## Description
A simple customizable training pipeline for semantic segmentation.
## Dataset
HierText dataset (https://github.com/google-research-datasets/hiertext).
## Usage
1. Convert dataset annotations to Run-length encoding (RLE) format
```python
python -m src.convert_dataset --dataset_dir ../dataset/hiertext --output_dir data
```
2. Train model
```python
python -m src.train --encoder_name resnet18 --dataset_dir ../dataset/hiertext --label_dir data --image_size 640 --batch_size 8 --num_workers 0 --lr 1e-4 --device 0 --amp --epochs 5
```
3. Inference

See [`inference_example.ipynb`](inference_example.ipynb)
