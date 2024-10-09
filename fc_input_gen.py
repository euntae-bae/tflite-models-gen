import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model-name', action='store', default='fc_triple', help='default=fc_triple')
parser.add_argument('--model-config', action='store', default='medium', help='default=medium')
parser.add_argument('--batch-size', '-b', type=int, action='store', default=1, help='default=1')
# parser.add_argument('--input-dim', action='store', help='default=12')
args = parser.parse_args()

# -3.0000에서 3.0000 사이의 랜덤 부동소수점 숫자 n_numbers개 생성
# FC basic, small
# DL_INPUT_DIM = 10

inputDim = 0
batchSize = 0

if args.model_name == 'fc_basic':
    inputDim = 10
    args.model_config = 'small'
    #batchSize = 1
elif args.model_name == 'fc_triple':
    if args.model_config == 'small':
        inputDim = 128
    elif args.model_config == 'med' or args.model_config == 'medium':
        inputDim = 512
    elif args.model_config == 'large':
        inputDim = 4096
    elif args.model_config == 'huge':
        inputDim = 16384

batchSize = args.batch_size

fileDir = 'fc_input'
fileName = f'{args.model_name}_{args.model_config}_{batchSize}'
filePath = fileDir + '/' + fileName + '.h'
fp = open(filePath, 'w')

print(f'model: {fileName}')
print(f'input dimension: {inputDim}, batch size: {batchSize}')

fp.write('// program generated\n')
fp.write(f'// model: {fileName}\n')
fp.write(f'// input dimension: {inputDim}, batch size: {batchSize}\n')

DL_INPUT_DIM = 128
DL_BATCH_SIZE = 512
n_numbers = DL_BATCH_SIZE * DL_INPUT_DIM
random_numbers = [ random.uniform(-3.0, 3.0) for _ in range(n_numbers) ]

# 결과 출력 (소수점 4자리까지 표시)
fp.write('float fc_input[%d] = { ' % (inputDim * batchSize))
i = 0
j = 0
ends = ', '
for n in random_numbers:
    if j == n_numbers - 1:
        ends = ' '
    fp.write("{0:.4f}".format(n) + ends)
    i += 1
    j += 1
    if i == DL_INPUT_DIM and j < n_numbers:
        fp.write('\n')
        i = 0
fp.write('};\n')
fp.close()
print(f'{filePath} is successfully generated')