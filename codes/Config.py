import torch
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import logging
# import transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch as con
logging.set_verbosity_error()


# con.convert_tf_checkpoint_to_pytorch(r'.\bert_model.ckpt',    r'.\bert_config.json',    r'.\pytorch_bert.bin')


# parameters configuration
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
WORKING_PATH = "../data"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = BertConfig.from_pretrained('../model')
# tokenizer = BertTokenizer.from_pretrained('../model')
model = BertModel.from_pretrained('bert-base-uncased')

train_fraction = 0.8
val_fraction = 0.1
batch_size = 32


# text encoder
d_model = 768  # Embedding Size
d_ff = 1024  # FeedForward dimension
d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 1  # number of heads in Multi-Head Attention
