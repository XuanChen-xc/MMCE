#!/bin/bash



######## Experiment 2: Get trends with varying lambda values ########
echo "Experiment 1 --------------------------------------------"
for i in 0.1 0.3 0.5 0.7 0.9 1.1 2.0
do
	echo 'Lambda' $i
	python entropy_loss.py --entropy_coeff=$i 

done
