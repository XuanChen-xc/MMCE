#!/bin/bash



######## Experiment : get different rotate angle mnist data ########
echo "Experiment 1 --------------------------------------------"
for ((i = 15; i <= 180; i = i + 15))
do
	echo 'Rotate Angle' $i
	python images.py --angle=$i

done
