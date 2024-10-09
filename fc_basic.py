import tensorflow as tf

# 단순한 FC 레이어 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),   # 입력 크기가 10인 벡터
    tf.keras.layers.Dense(5, activation='relu'),  # 출력 크기가 5인 FC 레이어
    tf.keras.layers.Dense(1, activation='sigmoid') # 이진 분류용 출력 레이어
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('fc_basic.tflite', 'wb') as f:
    f.write(tflite_model)
