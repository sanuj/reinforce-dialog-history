import torch
from parlai.core.params import ParlaiParser
from parlai.tasks.light_dialog.agents import DefaultTeacher
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel

parser = ParlaiParser(True, True, 'Evaluate a model')
DefaultTeacher.add_cmdline_args(parser)
args = ['-dt', 'valid']
opt = parser.parse_args(args, print_args=False)
# Opt({'datapath': 'data'})
teacher = DefaultTeacher(opt)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

for episode in teacher.episodes:
	for step in episode:
		context = tokenizer.encode(step['text'])
		final_response = None
		min_loss = float('inf')
		for cand in step['label_candidates']:
			response = tokenizer.encode(cand)
			conv = torch.tensor([context + response])
			loss, logits, _ = model(conv, labels=conv)
			if loss < min_loss:
				min_loss = loss
				final_response = response
		
		# print(step['text'])
		print(step['labels'])
		print(tokenizer.decode(final_response))
		print('-'*20)
