#!/bin/bash
python3 meta_featured_classifier.py >> NoElitism-4-fold.log
python3 meta_featured_classifier.py -e >> Elitism-4-fold.log
exit
