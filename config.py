""" Global configurations file.
"""

# Dataset settings
NUM_CLASSES = {
    'cifar10': 10,
    'cifar100':100,
    'imagenet': 1000,
    'cub200': 200,
    'cars': 196,
    'aircraft': 100,
}

DATA_FOLDERS = {
    'cifar': 'Path2DatasetCIFAR10/',
    'imagenet': 'Path2DatasetImageNet/',
    'cub200': 'Path2DatasetCUB_200_2011/',
    'cars': 'Path2DatasetStanfordCars/',
    'aircraft': 'Path2DatasetFGVCAircraft/',
}

MEANS = {
    'cifar': (0.4914, 0.4822, 0.4465),
    'imagenet': (0.485, 0.456, 0.406),
    'cub200': (0.485, 0.456, 0.406),
    'cars': (0.485, 0.456, 0.406),
    'aircraft': (0.485, 0.456, 0.406),
}

STDS = {
    'cifar': (0.2023, 0.1994, 0.2010),
    'imagenet': (0.229, 0.224, 0.225),
    'cub200': (0.229, 0.224, 0.225),
    'cars': (0.229, 0.224, 0.225),
    'aircraft': (0.229, 0.224, 0.225),
}

model_config={
    "gpt2":{
        "from_pretrained": "openai-community/gpt2",
        "attn_implementation": "flash_attention_2",
    },
    "Qwen_1.5b":{
        "from_pretrained": "Qwen/Qwen2-1.5B",
        "attn_implementation": "eager",

    },
    "Qwen_0.5b":{
        "from_pretrained": "Qwen/Qwen2-0.5B",
        "attn_implementation": "eager",
    }
}

# Model definition
TAU = 0.01
IS_TRAIN = True
K_LEVEL = 16
IS_NORMAL = True
IS_EMP = False
PRUNE = False
TOT_TRAIN_STEP = 0
PRUNE_END_STEP = 0
PRUNE_START_STEP = 0

# Training settings
BATCH_SIZE = {
    'cifar10': 128,
    'cifar100':128,
    'imagenet': 256,
    'cub200': 256,
    'cars': 256,
    'aircraft': 256,
}

EPOCH = {
    'cifar10': 350,
    'imagenet': 60,
    'cub200': 60,
    'cars': 60,
    'aircraft': 60,
}

LAYER = {
    'resnet20': 20,
    'resnet32': 32,
    'resnet56': 56,
    'vggsmall': 7,
    'resnet18': 21,
    'resnet50': 54,
    'mnasnet': 53,
    'proxylessnas': 62,
}

L_CNT = 0
LAYER_NUM = 20
EPS = 1e-8
KEEP = True
DEBUG = False
SKIPPED_LAYERS = []
INIT_METHOD = ""
PRUNE_SCALE = 0.1
PRUNE_FREQ = 10
DEBUG = False
SAMPLE = False
USE_AVERAGE = False
PRIOR = "spike_slab"

def set_status(flag):
    global IS_TRAIN
    IS_TRAIN = flag

def count_layer():
    global L_CNT
    L_CNT = L_CNT + 1


def set_config(args):
    global IS_EMP, IS_NORMAL, K_LEVEL, TAU, LAYER, LAYER_NUM, SKIPPED_LAYERS, INIT_METHOD, PRUNE, PRUNE_SCALE, PRUNE_FREQ, DEBUG, SAMPLE, USE_AVERAGE, PRIOR
    IS_EMP = args.empirical
    IS_NORMAL = args.normal
    TAU = args.tau
    K_LEVEL = args.K
    # LAYER_NUM = LAYER[args.network]
    # SKIPPED_LAYERS = [1, LAYER_NUM]
    INIT_METHOD = args.init_method
    PRUNE = args.prune
    PRUNE_SCALE = args.prune_scale
    # PRUNE_FREQ = args.prune_freq
    DEBUG=args.debug
    SAMPLE=args.sample
    USE_AVERAGE=args.average
    PRIOR = args.prior
    