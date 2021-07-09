#! /bin/bash
# Author: Knoxliu (dengkailiu@whu.edu.cn)
# All rights reserved.

# task identification, '0' refers to test model on eval dataset, '1' refres to test model both on develop dataset and evaluation dataset, and '2' refers to train model and test model subsequently.
stage=0

# key parameter
track=logical
dataset_type=LA
feature=wave
num_epochs=100
batch_size=128
learning_rate=0.0001
dataset_sz=20000
regular_str=${track}_${feature}_${num_epochs}_${batch_size}_${learning_rate}_${dataset_sz}

# model path
#model_path=./models/model_${regular_str}
# path to ASV scores towads multiple spoof methods
asv_score_path=../data_logical/LA/ASVspoof2019_LA_asv_scores

#model file with name 'epoch_*.pth'
#last_model=$(echo "$num_epochs - 1" | bc)
model_name=final.pth

#output cm score file path
cm_score_path=./res

info_dir=./temp_info_model_${regular_str}
if [ -d $info_dir ];then
	tips='process detail are saved in directory: '
	echo $tips
	echo $info_dir
else
	tips='process detail directory created successfully!'
	mkdir $info_dir
	echo $tips
	echo $info_dir
fi

if [ $stage -ge 2 ];then
	#python model_main.py --num(_epochs=$num_epochs --track=$track --feature=$feature --lr=$learning_rate --dataset_sz=$dataset_sz
	python model_main.py --batch_size=$batch_size --num_epochs=$num_epochs --track=$track --feature=$feature --lr=$learning_rate --dataset_sz=$dataset_sz > ${info_dir}/train_process.txt
fi

#exit

if [ $stage -ge 1 ];then
	#python model_main.py --eval --eval_output=res --model_path=${model_path}/${model_name} --dataset_sz=$dataset_sz
	python model_main.py --batch_size=$batch_size --eval --num_epochs=$num_epochs --track=$track --feature=$feature --eval_output=res --lr=$learning_rate --model_name=${model_name} --dataset_sz=$dataset_sz > ${info_dir}/dev_test.txt
	#python evaluate_tDCF_asvspoof19.py ${cm_score_path}/model_${regular_str}_dev_res ${asv_score_path}/ASVspoof2019.${dataset_type}.asv.dev.gi.trl.scores.txt
	python evaluate_tDCF_asvspoof19.py ${cm_score_path}/model_${regular_str}_dev_res ${asv_score_path}/ASVspoof2019.${dataset_type}.asv.dev.gi.trl.scores.txt > ${info_dir}/dev_res.txt
fi

#exit

if [ $stage -ge 0 ];then
	#python model_main.py --eval --eval_output=res --model_path=${model_path}/${model_name} --is_test --dataset_sz=$dataset_sz
	python model_main.py --batch_size=$batch_size --eval --num_epochs=$num_epochs --track=$track --feature=$feature --eval_output=res --lr=$learning_rate --model_name=${model_name} --is_test --dataset_sz=$dataset_sz > ${info_dir}/eval_test.txt
	#python evaluate_tDCF_asvspoof19.py ${cm_score_path}/model_${regular_str}_eval_res ${asv_score_path}/ASVspoof2019.${dataset_type}.asv.eval.gi.trl.scores.txt
	python evaluate_tDCF_asvspoof19.py ${cm_score_path}/model_${regular_str}_eval_res ${asv_score_path}/ASVspoof2019.${dataset_type}.asv.eval.gi.trl.scores.txt > ${info_dir}/eval_res.txt
fi
