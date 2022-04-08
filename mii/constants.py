import enum


#TODO naming..
class DeploymentType(enum.Enum):
    LOCAL = 1
    #expose GPUs
    AML_LOCAL = 2
    AML_ON_AKS = 3

TENSOR_PARALLEL_KEY = 'tensor_parallel'
PORT_NUMBER_KEY = 'port_number'

MII_CONFIGS_KEY = 'mii_configs'
MII_CONFIGS_DEFAULT = {TENSOR_PARALLEL_KEY: 1, PORT_NUMBER_KEY: 50050}


class Tasks(enum.Enum):
    TEXT_GENERATION = 1
    TEXT_CLASSIFICATION = 2
    QUESTION_ANSWERING = 3


TEXT_GENERATION_NAME = 'text-generation'
TEXT_CLASSIFICATION_NAME = 'text-classification'
QUESTION_ANSWERING_NAME = 'question-answering'


MODEL_NAME_KEY = 'model_name'
TASK_NAME_KEY = 'task_name'

ENABLE_DEEPSPEED_KEY = 'ds_optimize'

MII_CACHE_PATH = "MII_CACHE_PATH"
MII_CACHE_PATH_DEFAULT = "/tmp/mii_cache"

MII_DEBUG_MODE = "MII_DEBUG_MODE"
MII_DEBUG_MODE_DEFAULT = "0"

MII_DEBUG_DEPLOY_KEY = "MII_DEBUG_DEPLOY_KEY"

MII_DEBUG_BRANCH = "MII_DEBUG_BRANCH"
MII_DEBUG_BRANCH_DEFAULT = "main"


