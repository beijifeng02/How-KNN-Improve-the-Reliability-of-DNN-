export CUDA_VISIBLE_DEVICES=1

MODEL=("resnet18" "resnet50")

for model in "${MODEL[@]}"; do
  python main.py --model "${model}" --dataset "svhn"
done