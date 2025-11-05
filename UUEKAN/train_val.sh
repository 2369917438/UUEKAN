dataset=heus
input_size=256
python train.py --arch UUEKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset} --data_dir inputs --b 4 --epochs 1
python val.py --name ${dataset} --model UUEKAN

dataset=cvc
input_size=256
python train.py --arch UUEKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset} --data_dir inputs --b 4 --epochs 1
python val.py --name ${dataset} --model UUEKAN

dataset=busi
input_size=256
python train.py --arch UUEKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset} --data_dir inputs --b 4 --epochs 1
python val.py --name ${dataset} --model UUEKAN