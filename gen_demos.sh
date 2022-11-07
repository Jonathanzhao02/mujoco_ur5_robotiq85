#!/bin/bash

for i in {2001...2002}
do
    python gen_demo.py $i
    exit
done
