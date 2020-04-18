import torch
import os
import json
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence, PackedSequence
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPLIT = 'valid'
BASE_DIR = '.'

with open(os.path.join(BASE_DIR, SPLIT + '.json')) as f:
  data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.to(device)
bert.eval()

MAX_LEN = 512
def tokenize(inp):
  tokens = tokenizer.encode(inp, add_special_tokens=True)
  if len(tokens) > MAX_LEN:
    tokens = tokenizer.encode('[CLS]') + tokens[-MAX_LEN+1:]
  return tokens

pad_idx = tokenizer.encode('[PAD]')[0]
def pad(inp):
  lens = [[len(t) for t in d] for d in inp]
  max_len = max([max(l) for l in lens])
  padded = [torch.tensor([s + (max_len-lens[i][j]) * [pad_idx] for j, s in enumerate(d)]).to(device) for i, d in enumerate(inp)]
  return padded

  # inp: batch x turns x str_len
def embed(inp, mask=None, out_type='cls'):
  out, pooler_out = bert(inp, mask)
  if out_type == 'cls':
    return out[:, 0, :] # output of [CLS] as embedding.
  elif out_type == 'pooler':
    # Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function.
    return pooler_out

def dialog2inp(d):
  inp = [d['name'] + ' ' + d['desc']]
  inp.append(d['self_name'] + ' ' + d['self_persona'])
  inp.extend([c['partner_say'] for c in d['conv'] if 'partner_say' in c])
  return inp

def dialog2x(d, argx):
  x = [t[argx] for t in d['conv']]
  return x

def prepare(batch):
  dialogs = [dialog2inp(d) for d in batch]
  cands = [dialog2x(d, 'candidates') for d in batch]
  labels = [dialog2x(d, 'label') for d in batch]
  
  # pad labels and candidates
  labels = pad_sequence([torch.tensor([-1]*(len(d)-len(l)) + l).to(device) for d, l in zip(dialogs, labels)], batch_first=True, padding_value=-1)
  mask = (labels > -1).float()
  labels[labels < 0] = 0
  cands = [[['']*20]*(len(d)-len(c)) + c for d, c in zip(dialogs, cands)]
  
  tokens = [[tokenize(t) for t in d] for d in dialogs]
  cand_tokens = [[[tokenize(c) for c in t] for t in ct] for ct in cands]
  cand_tokens = [pad(c) for c in cand_tokens]
  tokens = pad(tokens)
  return tokens, cand_tokens, labels, mask

def batch(data, batch_size=3):
  for i in tqdm(range(0, len(data), batch_size)):
    yield prepare(data[i:i+batch_size])

lstm = torch.nn.LSTM(input_size=768, hidden_size=768).to(device)
loss_function = torch.nn.CrossEntropyLoss(reduction='none')
lr=0.005
epoch=10
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

for e in range(epoch):
  epoch_loss = []
  start = time.time()
  for b in batch(data):
    # tokens => batch x turns x sent
    # cand_tokesn => batch x turns x 20 x sent
    # labels => batch x turns
    tokens, cand_tokens, labels, loss_mask = b
    tokens_pack = pack_sequence(tokens, enforce_sorted=False)
    cand_packs = [pack_sequence(c, enforce_sorted=False) for c in cand_tokens]

    # only works if you pad with 0
    tokens_mask = (tokens_pack.data > pad_idx).int()
    cand_masks = [(cp.data > pad_idx).int() for cp in cand_packs]

    with torch.no_grad():
      tokens_emb = embed(tokens_pack.data, tokens_mask)
      cand_embs = [embed(cp.data, mask) for cp, mask in zip(cand_packs, cand_masks)]

    tokens_emb_pack = PackedSequence(tokens_emb, tokens_pack.batch_sizes, tokens_pack.sorted_indices, tokens_pack.unsorted_indices)
    cand_emb_packs = [PackedSequence(cand_embs[i], cp.batch_sizes, cp.sorted_indices, cp.unsorted_indices) for i, cp in enumerate(cand_packs)]
    
    lstm.zero_grad()
    out_pack, (ht, ct) = lstm(tokens_emb_pack)
    
    out_pad, out_pad_len = pad_packed_sequence(out_pack, batch_first=True)
    cand_pads = pad_sequence([pad_packed_sequence(ce, batch_first=True)[0] for ce in cand_emb_packs], batch_first=True, padding_value=pad_idx)

    y_scores = torch.sum(torch.mul(out_pad.unsqueeze(2), cand_pads), -1)
    loss = loss_function(y_scores.view(-1, 20), labels.view(-1))
    loss_mean = torch.sum(torch.mul(loss, loss_mask.view(-1))) / torch.sum(loss_mask)
    epoch_loss.append(loss_mean)
    loss_mean.backward()
    optimizer.step()

  end = time.time()
  print('Epoch done.', e, end-start, torch.mean(torch.tensor(epoch_loss)))

