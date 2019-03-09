# classtf

classtf is a Random Forest Classifier based on Tensorflow (TM)

## Usage
classtf.py [-h] [-r FILE] [-t FILE] [-x FILE] [-f] [--loss-treshold VALUE] [--train-timeout TIME_INTERVAL]  [-c TARGET_FEATURE_ID] [--ignore-features  [...]] [-d MODEL_DIR] [-n NUM_OF_TREES] [-b BATCH_SIZE] [--depth NUM_NODES] [-v]

###  Arguments  
##### **-h**, **--help**
&nbsp;&nbsp;&nbsp;&nbsp;show this help message and exit

##### **-r FILE**, **--run FILE**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the classifier is run using the dataset FILE as input

##### **-t FILE**, **--train FILE**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the classifier is trained using the dataset FILE

##### **-x FILE**, **--test FILE**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the classifier is tested using the dataset FILE

##### **-f**, **--feature-importance**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the importance of each feature is computed. Can be used only if both **--train** and **--test** options are specified

##### **--loss-treshold VALUE**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the training will stop when the loss changes between two cycles becomes smaller than **VALUE** (or when the training timeout expires). This option has effect only when the option **--train** is specified and is ignored otherwise. If not specified, the default value of 0.001 is used.

##### **--train-timeout TIME_INTERVAL**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified set the maximum execution time for the training process. **TIME_INTERVAL** must be a string representing a time interval. Allowed units are y (years), d (days), h (hours), m (minutes), s (seconds) [i.e. 1y2d13h20m13.3s]

_If not specified, no timeout is applied_

##### **-c TARGET_FEATURE_ID**, **--target TARGET_FEATURE_ID**
  &nbsp;&nbsp;&nbsp;&nbsp;Set the name or column index of feature used as target class during training and testing. If not specified, the last column in the dataset is used as default

##### **--ignore-features  [ ...]**
  &nbsp;&nbsp;&nbsp;&nbsp;List of features that should be ignored

##### **-d MODEL_DIR**, **--model-dir MODEL_DIR**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, the trained model is saved or restored from **MODEL_DIR**

##### **-n NUM_OF_TREES**, **--trees NUM_OF_TREES**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, set the number of generated trees to **NUM_OF_TREES**, _otherwisee fallback to the default value of 1000 trees_

##### **-b BATCH_SIZE**, **--batch-size BATCH_SIZE**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, set the size of the batch to to **BATCH_SIZE**, which is the number of object used at once during a training/test/run cycle. _The default value is 4096_

##### **--depth NUM_NODES**
  &nbsp;&nbsp;&nbsp;&nbsp;If specified, set the maximum number of nodes created by the model to **NUM_NODES**. _The default value is 10000_

##### **-v**, **--version**
  &nbsp;&nbsp;&nbsp;&nbsp;Print the program version and exit

### NOTES:
  You **MUST** specify at least one of the -r, -t or -x
  options

## Examples

  - Simple training and testing:
    ```classtf.py -t cat1.fits -x cat2.fitsi -c "Class"```

  - Train the classifier and save the trained model:
    ```classtf.py --train traincat.fits --model-dir ./mymodel/```

  - Load saved model and run it on a catalogl:
    ```classtf.py --model-dir ./mymodel --run mycat.fits```

  - Train the classifier with a timeout of 1 day and 6 hours
    ```classtf.py --train traincat.votable --train_timeout 1d3h```
    ```classtf.py --train traincat.votable --train_timeout 1.25d```
    ```classtf.py --train traincat.votable --train_timeout 30h```

  - Train the classifier with a loss treshold of 0.001:
    ```classtf.py --train train.csv --loss-treshold 0.001```

For more info import the classtf module in python and run
```
 >>> help(classtf)
```

