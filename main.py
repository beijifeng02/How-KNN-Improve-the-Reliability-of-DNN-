from datasets.utils import build_dataloader
from cache.utils import build_classifier


model = build_classifier("resnet50")
trainloader, testloader = build_dataloader()
train_feature, train_logits, train_labels = model.run_and_cache_outputs(trainloader)
test_feature, test_logits, test_labels = model.run_and_cache_outputs(testloader, train=False)