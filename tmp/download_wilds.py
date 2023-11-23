from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms


dataset = get_dataset(dataset="iwildcam", download=True)
dataset = get_dataset(dataset="rxrx1", download=True)
# dataset = get_dataset(dataset="poverty", download=True)
# dataset = get_dataset(dataset="globalwheat", download=True)