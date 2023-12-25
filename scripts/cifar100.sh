export CUDA_VISIBLE_DEVICES=1

MODEL=("resnet18" "resnet50" "resnet101")

for model in "${MODEL[@]}"; do
  python main.py --model "${model}" --dataset "cifar100"
done