import os

# read_file paths
class Config_path:
    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    current_path = os.path.join(parent_path, 'data')  # "data" is current folder

    wiki_train_path = os.path.join(current_path, 'annotated_wd_data_train.txt')
    wiki_dev_path = os.path.join(current_path, 'dev_combination_wiki.csv')  ###############
    wiki_test_path = os.path.join(current_path, 'annotated_wd_data_test.txt')

    train_raw_path = os.path.join(current_path, 'train_raw.csv')  ###############
    test_raw_path = os.path.join(current_path, 'test_raw.csv')  ###############

    train_q_space_path = os.path.join(current_path, 'train_q_space.csv')  ###################
    train_combination_path = os.path.join(current_path, 'train_combination.csv')
    test_combination_path = os.path.join(current_path, 'test_combination.csv')

    triple = os.path.join(current_path, 'wiki_triple_small.csv')

    mapping_path = os.path.join(current_path, 'mapping.txt')
    checkpoint_path = "checkpoint"


# output paths
class Config_output_path:
    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    current_path = os.path.join(parent_path, 'models')

    ner_data_path = os.path.join(parent_path, 'data')
    transformers_path = os.path.join(parent_path, 'models')


# parameters for Seq2Seq transformer model
class Hyparams_transformers:
    BATCH_SIZE = 128
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    LEARNING_RATE = 0.0005

    N_EPOCHS = 1000
    CLIP = 1