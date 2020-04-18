import os
import json
from parlai.core.params import ParlaiParser
from parlai.tasks.light_dialog.agents import DefaultTeacher

SPLIT = 'train'
BASE_DIR = '.'

parser = ParlaiParser(True, True, 'Evaluate a model')
DefaultTeacher.add_cmdline_args(parser)
args = ['-dt', SPLIT]
opt = parser.parse_args(args, print_args=False)
teacher = DefaultTeacher(opt)

def first_word(s):
	l = s.split()
	return '' if len(l) == 0 else l[0]

def extractTurn(t, step):
	found_partner = False
	for s in step['text'].split('\n'):
		fw = first_word(s)
		if fw.find('partner') != -1:
			found_partner = True
		if found_partner:
			if fw == '_partner_say':
				t['partner_say'] = s[len(fw):]
			elif fw == '_self_say':
				t['self_say'] = s[len(fw):]
			elif fw == '_partner_emote':
				t['partner_emote'] = s[len(fw):]
			elif fw == '_self_emote':
				t['self_emote'] = s[len(fw):]
			elif fw == '_partner_act':
				t['partner_act'] = s[len(fw):]
			elif fw == '_self_act':
				t['self_act'] = s[len(fw):]
	t['self_say'] = step['labels'][0]
	t['label'] = step['label_candidates'].index(step['labels'][0])
	t['candidates'] = step['label_candidates']


def extractEnv(d, step):
	for s in step['text'].split('\n'):
		fw = first_word(s)
		if fw == '_setting_name':
			d['name'] = s[len(fw):]
		elif fw == '_setting_desc':
			d['desc'] = s[len(fw):]
		elif fw == '_partner_name':
			d['partner_name'] = s[len(fw):]
		elif fw == '_self_name':
			d['self_name'] = s[len(fw):]
		elif fw == '_self_persona':
			d['self_persona'] = s[len(fw):]
		elif fw == '_object_desc':
			d['objects'].append(s[len(fw):])
		elif fw == '_partner_say':
			d['self_first'] = False
	if 'self_first' not in d:
		d['self_first'] = True


# {
# 	"name": "",
# 	"desc": "",
# 	"partner_name": "",
# 	"self_name": "",
# 	"self_persona": "",
# 	"objects": ["obj 1", "obj 2"],
# 	"self_first": True,
# 	"conv": [
# 		{
# 			"partner_say": "",
# 			"partner_act": "",
# 			"partner_emote": "",
# 			"self_say": "",
# 			"self_emote": "",
# 			"self_act": "",
# 			"candidates": [],
# 			"label": 1,
# 		}
# 	],
# 	"vec": {}
# }
def main():
	dialogs = []

	for i, episode in enumerate(teacher.episodes):
		if i%2 == 0:
			continue
		dialog = {'objects': [], 'conv': []}
		for j, step in enumerate(episode):
			if j == 0:
				extractEnv(dialog, step)
			dialog['conv'].append({})
			extractTurn(dialog['conv'][-1], step)
		dialogs.append(dialog)

	with open(os.path.join(BASE_DIR, SPLIT + '_half.json'), 'w') as outfile:
		print('total', len(dialogs))
		json.dump(dialogs, outfile)

if __name__ == '__main__':
	main()
