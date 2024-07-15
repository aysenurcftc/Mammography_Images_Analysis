
# Breast Cancer Classification




## Installation



```bash
  pip install -r requirements.txt
```
    
## Usage/Examples

```javascript


usage: main.py [-h] [--dataset {CBIS-DDSM,CLAHE_images,MBCD_Implant}] [--mammogram_type MAMMOGRAM_TYPE] [--model {cnn,vit}] [--run_mode {training,test}]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--max_epochs MAX_EPOCHS] [--verbose_mode VERBOSE_MODE] [--visualize VISUALIZE]

Training and testing a CNN model for different datasets.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {CBIS-DDSM,CLAHE_images,MBCD_Implant}
                        Dataset to be used
  --mammogram_type MAMMOGRAM_TYPE
                        Type of mammogram
  --model {cnn,vit}     Model to be used
  --run_mode {training,test}
                        Run mode: training or test
  --learning_rate LEARNING_RATE
                        Learning rate for the model
  --batch_size BATCH_SIZE
                        Batch size for training
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs
  --verbose_mode VERBOSE_MODE
                        Verbose mode
  --visualize VISUALIZE
                        Visualize dataset


```

