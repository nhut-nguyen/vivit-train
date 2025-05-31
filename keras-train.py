import os
import numpy as np
import tensorflow as tf
import keras
import cv2
from keras import layers, ops
import io
import imageio

# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

# DATA
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (32, 56, 56, 3) #frame, height, width, color chanel
NUM_CLASSES = 11

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label

def convert_label(group):
    mapping = {
        1: 1,
        2: 2, 3: 2,
        4: 3, 5: 3, 14: 3, 23: 3, 25: 3, 40: 3,
        6: 4, 9: 4, 22: 4,
        11: 5,
        12: 6, 24: 6, 27: 6, 30: 6, 34: 6,
        16: 7, 17: 7, 21: 7, 31: 7,
        33: 8,
        36: 9,
        41: 10
    }
    return mapping.get(group, 0)  # 0 nếu không tìm thấy

def get_label_from_filename(filename):
    # Bỏ phần mở rộng .mp4
    base_name = os.path.splitext(filename)[0]

    # Tách chuỗi dựa trên dấu gạch dưới (_) và lấy phần đầu (nhãn)
    label = convert_label(int(base_name.split('_')[0]))

    return int(label)  # Trả về nhãn dạng số nguyên


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("Prepare DataLoader")
    return dataloader

# Hàm đọc và xử lý video
def load_video(video_path, num_frames, target_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    print("Load Video:", video_path)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Thay đổi kích thước của từng frame
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()

    # Nếu số khung hình ít hơn yêu cầu, thêm các khung hình đen
    if len(frames) < num_frames:
        frames.extend([np.zeros(target_size + (3,), dtype=np.uint8)] * (num_frames - len(frames)))

    # Chuyển đổi danh sách frames thành ndarray
    return np.array(frames)


def data_to_tensor(video_dir):

    # Số lượng khung hình bạn muốn trích xuất từ mỗi video
    num_frames = INPUT_SHAPE[0]
    target_size = (INPUT_SHAPE[1], INPUT_SHAPE[2])  # Kích thước khung hình (height, width)

    # Khởi tạo danh sách lưu video và nhãn
    videos = []
    video_labels = []

    # Duyệt qua từng file video trong thư mục
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)

            # Load video và chuyển đổi thành ndarray
            video_data = load_video(video_path, num_frames, target_size)

            # Lưu video và nhãn tương ứng
            videos.append(video_data)
            video_labels.append(get_label_from_filename(video_file))

    # Chuyển đổi danh sách video và nhãn thành ndarray
    video_tensor = np.array(videos)  # (num_samples, num_frames, height, width, channels)
    label_tensor = np.array(video_labels)  # (num_samples,)
    print("Data To Tensor: ", video_dir)
    return (video_tensor, label_tensor)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

#Using Model 1: Spatio-temporal attention
def VivitSpatioTemporalAttentionModel(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=ops.gelu),
                layers.Dense(units=embed_dim, activation=ops.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    print("Create Model ViViT Spatio-Temporal Attention")

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def RunExperiment():
    # Initialize model
    model = VivitSpatioTemporalAttentionModel(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    print("Training Model")
    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)

    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model

(train_videos, train_labels) = data_to_tensor('D:/Projects/autism/vivit_model_python/data/train/')
(valid_videos, valid_labels) = data_to_tensor('D:/Projects/autism/vivit_model_python/data/valid/')
(test_videos, test_labels) = data_to_tensor('D:/Projects/autism/vivit_model_python/data/test/')

trainloader = prepare_dataloader(train_videos, train_labels, "train")
validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")

model = RunExperiment()
model.save('vivit_model.keras')

#Inference Model

NUM_SAMPLES_VIZ = 250
testsamples, labels = next(iter(testloader))
testsamples, labels = testsamples[:NUM_SAMPLES_VIZ], labels[:NUM_SAMPLES_VIZ]

ground_truths = []
preds = []
videos = []
model.summary()

for i, (testsample, label) in enumerate(zip(testsamples, labels)):
    # Generate gif
    testsample = np.reshape(testsample.numpy(), INPUT_SHAPE)
    with io.BytesIO() as gif:
        imageio.mimsave(gif, (testsample * 255).astype("uint8"), "GIF", fps=5)
        videos.append(gif.getvalue())

    # Get model prediction
    output = model.predict(ops.expand_dims(testsample, axis=0))[0]
    pred = np.argmax(output, axis=0)

    ground_truths.append(label.numpy().astype("int"))
    preds.append(pred)

boxes = []
all_label = [0, 1]

for i in range(NUM_SAMPLES_VIZ):
    true_class_i = ground_truths[i]
    pred_class_i = preds[i]
    print(f"T: {true_class_i} | P: {pred_class_i}")

