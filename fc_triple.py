import tensorflow as tf
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', action='store', help='small, medium, large, xl, xxl, huge; (default=medium)')
args = parser.parse_args()

inputSize = 0
outputFC1 = 0
outputFC2 = 0
outputFC3 = 0
modelName = 'fc_triple'

if args.model == 'small':
    inputSize = 128
    outputFC1 = 64
    outputFC2 = 10
    outputFC3 = 1
    modelName += '_small'
elif args.model == 'large':
    inputSize = 4096
    outputFC1 = 1024
    outputFC2 = 256
    outputFC3 = 1
    modelName += '_large'
elif args.model == 'xl':
    inputSize = 8192
    outputFC1 = 2048
    outputFC2 = 512
    outputFC3 = 1
    modelName += '_xl'
elif args.model == 'xxl':
    #inputSize = 12000
    inputSize = 11000
    outputFC1 = 2048
    outputFC2 = 512
    outputFC3 = 1
    modelName += '_xxl'
elif args.model == 'huge':
    inputSize = 16384
    outputFC1 = 4096
    outputFC2 = 1024
    outputFC3 = 1
    modelName += '_huge'
elif args.model == 'med' or args.model == 'medium' or args.model == None:
    inputSize = 512
    outputFC1 = 128
    outputFC2 = 64
    outputFC3 = 1
    modelName += '_medium'
else:
    print('E: %s is not available' % args.model)
    exit(1)

print('Model configuration: %s' % args.model)
print(f'FC1: {inputSize}x{outputFC1}')
print(f'FC2: {outputFC1}x{outputFC2}')
print(f'FC3: {outputFC2}x{outputFC3}')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(inputSize,)),
    tf.keras.layers.Dense(outputFC1, activation='relu'),
    tf.keras.layers.Dense(outputFC2, activation='softmax'),
    tf.keras.layers.Dense(outputFC3, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

modelDir = 'models'
modelName += '.tflite'
with open(modelDir + '/' + modelName, 'wb') as f:
    f.write(tflite_model)
    print(f'The TFLite model is successfully saved to {modelName}')

