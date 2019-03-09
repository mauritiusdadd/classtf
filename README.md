# classtf

classtf is a Random Forest Classifier based on Tensorflow (TM)

usage: classtf.py [-h] [-r FILE] [-t FILE] [-x FILE] [-f]
                  [--loss-treshold VALUE] [--train-timeout TIME_INTERVAL]
                  [-c TARGET_FEATURE_ID] [--ignore-features  [...]]
                  [-d MODEL_DIR] [-n NUM_OF_TREES] [-b BATCH_SIZE]
                  [--depth NUM_NODES] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -r FILE, --run FILE   If specified, the classifier is run using the dataset
                        FILE as input
  -t FILE, --train FILE
                        If specified, the classifier is trained using the
                        dataset FILE
  -x FILE, --test FILE  If specified, the classifier is tested using the
                        dataset FILE
  -f, --feature-importance
                        If specified, the importance of each feature is
                        computed. Can be used only if both --train and --test
                        options are specified
  --loss-treshold VALUE
                        If specified, the training will stop when the loss
                        changes between two cycles becomes smaller than VALUE
                        (or when the training timeout expires). This option
                        has effect only when the option --train is specified
                        and is ignored otherwise. If not specified, the
                        default value of 0.001 is used.
  --train-timeout TIME_INTERVAL
                        If specified set the maximum execution time for the
                        training process. TIME_INTERVAL must be a string
                        representing a time interval. Allowed units are y
                        (years), d (days), h (hours), m (minutes), s (seconds)
                        [i.e. 1y2d13h20m13.3s If not specified, no timeout is
                        applied
  -c TARGET_FEATURE_ID, --target TARGET_FEATURE_ID
                        Set the name or column index of feature used as target
                        class during training and testing. If not specified,
                        the last column in the dataset is used as default
  --ignore-features  [ ...]
                        List of features that should be ignored
  -d MODEL_DIR, --model-dir MODEL_DIR
                        If specified, the trained model is saved or restored
                        from MODEL_DIR
  -n NUM_OF_TREES, --trees NUM_OF_TREES
                        If specified, set the number of generated trees to
                        NUM_OF_TREES. Otherwisee fallback to the default value
                        of 1000 trees
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        If specified, set the size of the batch to to
                        BATCH_SIZE, which is the number of object used at once
                        during a training/test/run cycle. The default value is
                        4096
  --depth NUM_NODES     If specified, set the maximum number of nodes created
                        by the model to NUM_NODES. The default value is 10000
  -v, --version         Print the program version and exit

Notes:
  You *MUST* specify at least one of the -r, -t or -x
  options

Examples:

  - Simple training and testing:
    classtf.py -t cat1.fits -x cat2.fitsi -c "Class"

  - Train the classifier and save the trained model:
    classtf.py --train traincat.fits --model-dir ./mymodel/

  - Load saved model and run it on a catalogl:
    classtf.py --model-dir ./mymodel --run mycat.fits

  - Train the classifier with a timeout of 1 day and 6 hours
    classtf.py --train traincat.votable --train_timeout 1d3h
    classtf.py --train traincat.votable --train_timeout 1.25d
    classtf.py --train traincat.votable --train_timeout 30h

  - Train the classifier with a loss treshold of 0.001
    classtf.py --train train.csv --loss-treshold 0.001

For more info import the classtf module in python and run
>>> help(classtf)
