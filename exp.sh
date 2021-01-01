#!/usr/bin/env bash
for g in 0 1 2 3;
do
    for l in 1 2 3 4 5;
    do
        python learn.py -g ${g} -e ${g} -l ${l} &
    done
done
