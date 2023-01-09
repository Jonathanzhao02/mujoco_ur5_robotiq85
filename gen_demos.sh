#!/bin/bash

for i in {1..4000}
do
    python gen_demo.py $i $1
done
