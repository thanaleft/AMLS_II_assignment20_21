import string
import collections
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import re
from tqdm import tqdm
import numpy as np
import random
import time
import scikitplot as skplt
import torch.nn.functional as F
import torch.nn as nn
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, accuracy_score, f1_score
import torch.nn as nn

class BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, weight):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        embedding = nn.Embedding.from_pretrained(weight)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 4, 2)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 4, hidden_dim * 4))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 4, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x1, x2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        embedding_text = self.dropout(self.embedding(x1)) #[seq_len, batch, embedding_dim]
        #embedding_text = embedding_text + (0.2**0.5)*torch.randn(embedding_text.shape,device=device)
        
        embedding_topic = self.dropout(self.embedding(x2))
        #embedding_topic = embedding_topic + (0.2**0.5)*torch.randn(embedding_topic.shape,device=device)

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        text_output, (final_hidden_state, final_cell_state) = self.rnn(embedding_text)
        text_output = text_output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]
        text_output = text_output + (0.2**0.5)*torch.randn(text_output.shape,device=device)

        
        topic_output, (final_hidden_state1, final_cell_state1) = self.rnn(embedding_topic)
        topic_output = topic_output.permute(1, 0, 2)
        topic_output = topic_output + (0.2**0.5)*torch.randn(topic_output.shape,device=device)
        
        topic_mean = torch.mean(topic_output,dim=1)
        topic_mean = torch.unsqueeze(topic_mean,dim=1).expand(text_output.shape)
        
        lstm_output = torch.cat((text_output,topic_mean),dim=2)
        
        attn_output = self.attention_net(lstm_output)
        
        
        logit = self.fc(attn_output)
        return logit
    
    
# Define LSTM Tokenizer
def tokenizer_lstm(X, vocab, seq_len, padding):
    '''
    Returns tokenized tensor with left/right padding at the specified sequence length
    '''
    X_tmp = np.zeros((len(X), seq_len), dtype=np.int64)
    for i, text in enumerate(X):
        tokens = tokenize_text(text, 3) 
        token_ids = [vocab[word] for word in tokens if word in vocab.keys()]
        end_idx = min(len(token_ids), seq_len)
        if padding == 'right':
            X_tmp[i,:end_idx] = token_ids[:end_idx]
        elif padding == 'left':
            start_idx = max(seq_len - len(token_ids), 0)
            X_tmp[i,start_idx:] = token_ids[:end_idx]

    return torch.tensor(X_tmp, dtype=torch.int64)

def encoded_label(sentiment):
    if sentiment == 'negative':
        return 0
    else:
        return 1

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

#Define the tokenzation function
def tokenize_text(text, option):
    '''
    Tokenize the input text as per specified option
        1: Use python split() function
        2: Use regex to extract alphabets plus 's and 't
        3: Use ekphrasis text_processor.pre_process_doc
        4: Use NLTK word_tokenize(), remove stop words and apply lemmatization
    '''
    if option == 1:
        return text.split()
    elif option == 2:
        return re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text)
    elif option == 3:
        return [word for word in text_processor.pre_process_doc(text) if (word!='s' and word!='\'')]
    elif option == 4:
        words = [word for word in word_tokenize(text) if (word.isalpha()==1)]
        # Remove stop words
        stop = set(stopwords.words('english'))
        words = [word for word in words if (word not in stop)]
        # Lemmatize words (first noun, then verb)
        wnl = nltk.stem.WordNetLemmatizer()
        lemmatized = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in words]
        return lemmatized
    else:
        print("Please specify option value between 1 and 4")
        return []

# Define a DataSet Class which simply return (x, y) pair
class SimpleDataset(Dataset):
    def __init__(self, x, y, z):
        self.datalist=[(x[i], y[i], z[i]) for i in range(len(y))]
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self,idx):
        return self.datalist[idx]

# Data Loader
def create_data_loader(X, Y, z, batch_size, shuffle):
    X_sampled = np.array(X, dtype=object)
    Y_sampled = np.array(Y, dtype=object)
    z_sampled = np.array(z).astype(int)
    dataset = SimpleDataset(X_sampled, Y_sampled, z_sampled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# Define metrics
def metric(y_true, y_pred):
    rec = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return rec, acc, f1

def run():
    model_path = 'model/modelB.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path,map_location=device)
    model = model.to(device)
    model.eval()

    data_path = 'Dataset/B/test.txt'
    # Read data from txt file
    data_df = pd.read_table(data_path,sep='\t',header=None)
    # data_df = data_df.drop(columns=3)
    data_df.columns = ['ID','Topic','Sentiment','Text']
    data_df['label'] = data_df.Sentiment.apply(encoded_label)
    word2idx = {}
    file = open('Dataset/B/word2idxB.txt','r', encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        k = line.split('\t')[0]
        v = line.split('\t')[1]
        word2idx[k] = v
    file.close()
    testloader = create_data_loader(data_df['Text'], data_df['Topic'], data_df['label'],64,False)
    seq_len=33
    batch_size=64
    y_truth_tmp, y_pred_tmp = [], []

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            text_batch, topic, labels = batch
            # Skip the last batch of which size is not equal to batch_size
            if labels.size(0) != batch_size:
                break

            # Tokenize the input and move to device
            text_batch = tokenizer_lstm(text_batch, word2idx, seq_len, padding='left').transpose(1,0).to(device)
            topic = tokenizer_lstm(topic, word2idx, 4, padding='left').transpose(1,0).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device)

            # Get output and hidden state from the model, calculate the loss
            logits = model(text_batch, topic)

            y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
            y_truth_tmp.extend(labels.cpu().numpy())
        rec, acc, f1 = metric(y_truth_tmp, y_pred_tmp)
        print("Result for task B:AvgRec:{:.4f}, Acc: {:.4f}, F1: {:.4f}".format(rec, acc, f1))