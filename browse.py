from parlai.core.params import ParlaiParser
from parlai.tasks.light_dialog.agents import DefaultTeacher

DATA = 'valid'
DIAL_NO = 499

parser = ParlaiParser(True, True, 'Evaluate a model')
DefaultTeacher.add_cmdline_args(parser)
args = ['-dt', DATA]
opt = parser.parse_args(args, print_args=False)
# Opt({'datapath': 'data'})
teacher = DefaultTeacher(opt)

episode = teacher.episodes[DIAL_NO]
for step in episode:
	print(step['text'])
	print('label -->', step['labels'])
	print('-'*20)
