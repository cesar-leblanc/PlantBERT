#!/bin/bash

source ~/environments/pl@ntbert/bin/activate

accelerate launch --multi_gpu main.py "$@"