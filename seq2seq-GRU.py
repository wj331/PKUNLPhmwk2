from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import argparse, time
import numpy as np

from torch import Tensor
import random


# 加载数据 (to load the data)
def load_data(num_train):
    zh_sents = {}
    en_sents = {}
    for split in ['train', 'val', 'test']:
        zh_sents[split] = []
        en_sents[split] = []
        with open(f"./data/zh_en_{split}.txt", encoding='utf-8') as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh = zh.split()
                en = en.split()
                zh_sents[split].append(zh)
                en_sents[split].append(en)
    num_train = len(zh_sents['train']) if num_train==-1 else num_train
    zh_sents['train'] = zh_sents['train'][:num_train]
    en_sents['train'] = en_sents['train'][:num_train]
    print("训练集 验证集 测试集大小分别为", len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents

# 构建词表 (to build the vocabulary, map word to index)
class Vocab():
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = []
        self.add_word("[BOS]")
        self.add_word("[EOS]")
        self.add_word("[UNK]")
        self.add_word("[PAD]")
    
    def add_word(self, word):
        """
        将单词word加入到词表中
        """
        if word not in self.word2idx:
            self.word2cnt[word] = 1
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1
    
    def add_sent(self, sent):
        """
        将句子sent中的每一个单词加入到词表中
        sent是由单词构成的list
        """
        for word in sent:
            self.add_word(word)
    
    def index(self, word):
        """
        若word在词表中则返回其下标，否则返回[UNK]对应序号
        """
        #return index of the word else return index of unknown if word not in dictionary
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent, max_len):
        """
        在句子sent的首尾分别添加BOS和EOS之后编码为整数序列
        """
        #returns the index of sentence in dictionary
        encoded = [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
        return encoded
    
    def decode(self, encoded, strip_bos_eos_pad=False):
        """
        将整数序列解码为单词序列
        """
        #returns the word of sentence in dictionary
        return [self.idx2word[_] for _ in encoded if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]]
    
    def __len__(self):
        """
        返回词表大小
        """
        return len(self.idx2word)
    

# 定义模型 (base GRU model)
class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Gates: Update and Reset
        self.weight_ih = nn.Linear(input_size, 2 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 2 * hidden_size)

        #Candidate hidden state
        self.weight_candidate = nn.Linear(input_size, hidden_size)
        self.weight_hh_candidate = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hx):
        """
        input: (batch_size, input_size)
        hx: (batch_size, hidden_size)
        """
        # Compute gates
        gates = self.weight_ih(input) + self.weight_hh(hx)
        z_t, r_t = torch.split(torch.sigmoid(gates), self.hidden_size, dim=1)  # Split into update/reset gates
        
        # Compute candidate hidden state
        candidate = torch.tanh(self.weight_candidate(input) + r_t * self.weight_hh_candidate(hx))
        
        # Update hidden state
        hx = (1 - z_t) * hx + z_t * candidate
        return hx
    
# EncoderRNN
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.gru = GRUCell(embedding_dim, hidden_size)
        
    
    def forward(self, input, hidden):
        """
        input: N
        hidden: N * H
        
        输出更新后的隐状态hidden（大小为N * H）
        """
        # seq_len, batch_size = input_seq.size()
        # outputs = torch.zeros(seq_len, batch_size, self.hidden_size, device=input_seq.device)

        # embedded = self.embedding(input)

        # for t in range(seq_len):
        #     hidden = self.gru(embedded[t], hidden)
        #     outputs[t] = hidden
        # return outputs, hidden
        embedded = self.embedding(input)
        hidden = self.gru(embedded, hidden)
        return hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden, encoder_outputs):
        """
        hidden: N * H
        encoder_outputs: S * N * H
        
        输出attention加权后的context（大小为N * H）
        """
        seq_len, batch_size, hidden_size = encoder_outputs.size()
        hidden = hidden.unsqueeze(0).expand(seq_len, -1, -1)  # S * N * H
        score = self.v(torch.tanh(self.W(hidden) + self.U(encoder_outputs)))  # S * N * 1
        attention_weights = F.softmax(score, dim=0)  # S * N * 1
        context = torch.sum(attention_weights * encoder_outputs, dim=0)  # N * H
        return context, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout = 0.1):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.gru = GRUCell(embedding_dim, hidden_size)

        #add dropout
        self.dropout = nn.Dropout(dropout)

        #add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
        # self.h2o = nn.Linear(hidden_size, vocab_size) //replaced with self.out
        self.softmax = nn.LogSoftmax(dim=1)

        self.attention = BahdanauAttention(self.hidden_size)
        self.attention_combine = nn.Linear(hidden_size + embedding_dim, embedding_dim)
        
        #multiple layers with activation
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, input, hidden, encoder_outputs):
        """
        input: N
        hidden: N * H
        
        输出对于下一个时间片的预测output（大小为N * V）更新后的隐状态hidden（大小为N * H）
        """
        embedding = self.embedding(input)
        embedding = self.dropout(embedding)

        context, attention_weights = self.attention(hidden, encoder_outputs)

        #combine context and embedding
        gru_input = torch.cat((embedding, context), dim=1)
        gru_input = self.attention_combine(gru_input)
        gru_input = F.relu(gru_input)

        #GRU step
        hidden = self.gru(gru_input, hidden)
        hidden = self.layer_norm(hidden)

        #output projection
        output = self.out(hidden)
        output = self.softmax(output)
        return output, hidden, attention_weights

# Add label smoothing loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return self.criterion(x, true_dist.detach())

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len):
        super(Seq2Seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderRNN(len(tgt_vocab), embedding_dim, hidden_size)
        self.max_len = max_len
        
    def init_hidden(self, batch_size):
        """
        初始化编码器端隐状态为全0向量（大小为1 * H）
        """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size):
        """
        预测时，初始化解码器端输入为[BOS]（大小为batch_size）
        """
        device = next(self.parameters()).device
        return (torch.ones(batch_size)*self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src):
        """
        src: N * L
        编码器前向传播，输出最终隐状态hidden (N * H)和隐状态序列encoder_hiddens (N * L * H)
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        # 编码器端每个时间片，取出输入单词的下标，与上一个时间片的隐状态一起送入encoder，得到更新后的隐状态，存入enocder_hiddens
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        encoder_hiddens = torch.stack(encoder_hiddens, dim=0)
        return hidden, encoder_hiddens
    
    def forward_decoder(self, tgt, hidden, encoder_hiddens, tfr):
        """
        tgt: N
        hidden: N * H
        
        解码器前向传播，用于训练，使用teacher forcing，输出预测结果outputs，大小为N * L * V，其中V为目标语言词表大小
        """
        Bs, Lt = tgt.size()
        outputs = []
        input = tgt[:, 0]

        for i in range(Lt):
            output, hidden, attention_weights = self.decoder(input, hidden, encoder_hiddens)
            outputs.append(output)

            use_teacher_forcing = random.random() < tfr
            if use_teacher_forcing and i < Lt-1:
                input = tgt[:, i+1]
            else:
                input = output.argmax(-1)
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    
    def forward(self, src, tgt, tfr):
        """
            src: 1 * Ls
            tgt: 1 * Lt
            
            训练时的前向传播
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens, tfr)
        return outputs
    
    def predict(self, src):
        """
            src: 1 * Ls
            
            用于预测，解码器端初始输入为[BOS]，之后每个位置的输入为上个时间片预测概率最大的单词
            当解码长度超过self.max_len或预测出了[EOS]时解码终止
            输出预测的单词编号序列，大小为1 * L，L为预测长度
        """
        hidden, encoder_hiddens = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = []
        while len(preds) < self.max_len:
            output, hidden, attention_weights = self.decoder(input, hidden, encoder_hiddens)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        preds = torch.stack(preds, dim=-1)
        return preds
    
# 构建Dataloader
def collate(data_list):
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt

def padding(inp_ids, max_len, pad_id):
    max_len += 2    # include [BOS] and [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_

def create_dataloader(zh_sents, en_sents, max_len, batch_size, pad_id):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split=='train' else False
        datas = [(padding(zh_vocab.encode(zh, max_len), max_len, pad_id), padding(en_vocab.encode(en, max_len), max_len, pad_id)) for zh, en in zip(zh_sents[split], en_sents[split])]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

# 训练模型
def train_loop(model, optimizer, criterion, loader, device, current_epoch, max_epochs):
    model.train()
    epoch_loss = 0.0

    #add label smoothing
    smoothing = 0.1
    criterion = LabelSmoothingLoss(size=model.decoder.vocab_size, smoothing=smoothing)

    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tfr = max(0.4, 1.0 - current_epoch/max_epochs)    # teacher forcing ratio
        outputs = model(src, tgt, tfr)
        loss = criterion(outputs[:,:-1,:].reshape(-1, outputs.shape[-1]), tgt[:,1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)     # 裁剪梯度，将梯度范数裁剪为1，使训练更稳定
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    return epoch_loss

def test_loop(model, loader, tgt_vocab, device):
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].unsqueeze(0).to(device)     # 1 * L
            with torch.no_grad():
                outputs = model.predict(_src)         # 1 * L
            
            # 保留预测结果，使用词表vocab解码成文本，并删去BOS与EOS
            ref = " ".join(tgt_vocab.decode(tgt[_].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            references.append(ref)    # 标准答案
            hypotheses.append(hypo)   # 预测结果
    
    score = bleu.corpus_score(hypotheses, [references]).score      # 计算BLEU分数
    return hypotheses, references, score

# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()      
    parser.add_argument('--num_train', default=-1, help="训练集大小，等于-1时将包含全部训练数据")
    parser.add_argument('--max_len', default=10, help="句子最大长度")
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--num_epoch', default=10)
    parser.add_argument('--lr', default=0.0005)
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print("中文词表大小为", len(zh_vocab))
    print("英语词表大小为", len(en_vocab))

    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, pad_id=zh_vocab.word2idx['[PAD]'])

    torch.manual_seed(1)
    #Use GPU for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    if torch.cuda.is_available():
        print("GPU: ", torch.cuda.get_device_name(0))

    #initialise Model
    model = Seq2Seq(zh_vocab, en_vocab, embedding_dim=256, hidden_size=256, max_len=args.max_len)
    model.to(device)
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) #introduce more parameters for Adam
    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0 # set the loss of [PAD] to zero
    criterion = nn.NLLLoss(weight=weights)

    # #introduce scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=2,
    #     verbose=True
    # )

    # 训练
    start_time = time.time()
    best_score = 0.0
    best_epoch = 0
    
    for epoch in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device, epoch, args.num_epoch)
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device)
        # scheduler.step(bleu_score)
        # 保存验证集上bleu最高的checkpoint
        if bleu_score > best_score:
            torch.save(model.state_dict(), "model_best.pt")
            best_score = bleu_score
            best_epoch = epoch
        print(f"Epoch {epoch}: loss = {loss}, valid bleu = {bleu_score}")
        print(references[0])
        print(hypotheses[0])
    end_time = time.time()

    #测试
    model.load_state_dict(torch.load("model_best.pt"))
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])
    print(f"Training time: {round((end_time - start_time)/60, 2)}min")