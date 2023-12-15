# CV-11 기초대회 베이스라인 코드

## Project Structure

```
${PROJECT}
├── data
|   ├── eval
|   ├── train
├── model
├── output
├── loss.py
├── model.py
├── dataset.py
├── train.py
├── inference.py
├── README.md
└── requirements.txt
```

- dataset.py : This file contains dataset class for model training and validation
- inference.py : This file used for predict the model
- loss.py : This file defines the loss functions used during training
- model.py : This file defines the model
- README.md
- requirements.txt : contains the necessary packages to be installed
- train.py : This file used for training the model

## Getting Started

### Install Requirements

To Insall the necessary packages liksted in `requirements.txt`, run the following command while your virtual environment is activated:


```
pip install -r requirements.txt
```

### Usage

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.

```
python train.py
```

#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.

```
python inference.py --model BaseModel --model_dir ./model/exp
```

#### Tensorboard
```
tensorboard --logdir=save_dir
```

#### WandB
```
pip install wandb
wandb login
```