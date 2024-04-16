


set -x

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=esc50
imagenetpretrain=True
audiosetpretrain=True
bal=none
# 定义可能的超参数值的数组
lr_options=(1e-5 1.25e-5 1.5e-5 0.75e-5 1.35e-5 1.6e-5 1.8e-5 2-e5 0.8e-5 0.9e-5 )
batch_size_options=(44 46 48 44 50 52 42 40 38 36)
lrscheduler_decay_options=(0.85 0.88 0.9 0.92 0.83 0.8)
# 从这些数组中随机选取一个值

lr=${lr_options[$RANDOM % ${#lr_options[@]}]}
#batch_size=${batch_size_options[$RANDOM % ${#batch_size_options[@]}]}
lrscheduler_decay=${lrscheduler_decay_options[$RANDOM %{#lrscheduler_decay_options[@]}]}

# 打印选取的值
echo "Selected learning rate: $lr"
#echo "Selected batch size: $batch_size"
echo "Selected decay: $lrscheduler_decay"

#if [ $audiosetpretrain == True ]
#then
 # lr=1.5e-5
#else
 # lr=1e-4
#fi

freqm=24
timem=96
mixup=0
epoch=25
batch_size=48
fstride=10
tstride=10

dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
#lrscheduler_decay=0.9

base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}-lrdecay${lrscheduler_decay}

python ./prep_esc50.py


folders=("fold1" "fold2" "fold3" "fold4" "fold5")

# 定义一个标志变量，用来判断是否所有的fold都存在
all_exist=true

# 使用一个循环来检查每个文件夹是否存在
for folder in "${folders[@]}"; do
    if [ ! -d "$base_exp_dir/$folder" ]; then    # 如果文件夹不存在
        all_exist=false                          # 把标志变量设置成false
        break                                    # 并且退出循环
    fi
done

if $all_exist ; then
    echo 'exp exist'
    exit
else
    rm -rf "$base_exp_dir"         # 删除base_exp_dir
    mkdir -p "$base_exp_dir"       # 重新创建base_exp_dir
fi


for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}
done

#python ./get_esc_result.py --exp_path ${base_exp_dir}