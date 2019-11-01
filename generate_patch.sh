for subset in `seq 0 9`;
do
python -W ignore infinite_generator_2D.py --fold $subset --scale 32 --data datasets/thyroid/ --save datasets/thyroid/generated_patch
done
