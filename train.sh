export CUDA_VISIBLE_DEVICES=0

python -u train.py -y yamls/resnet_18.yaml
# nohup python -u train.py -y yamls/resnet_18.yaml &> logs/r18_s2_static.log &
#python train.py -y yamls/resnet_18.yaml
