# XÂY DỰNG HỆ THỐNG HỖ TRỢ SÀN LỌC TRẺ TỰ KỶ THÔNG QUA ĐẶC TRƯNG HÀNH VI

**Nhựt Nguyễn**  
Email: [nhut.works@gmail.com](mailto:nhut.works@gmail.com)  
Model: Video Vision Transformer - ViViT  
Language: Python

## Tập dữ liệu
URL: [Google Drive Dataset](https://drive.google.com/drive/folders/12MA19bjXp0d863bhCC8f73KnLtmsjghn?usp=drive_link)

## Nhãn
| #  | Nhóm hành vi        | Nhãn                   | Mô tả             |
|----|----------------------|------------------------|-------------------|
| 1  | vo_tay              | 1                      |                   |
| 2  | lac_lu_quay_tron    | 2, 3                   |                   |
| 3  | cu_dong_tay         | 4, 5, 14, 23, 25, 40    |                   |
| 4  | choi_do_vat         | 6, 9, 22               |                   |
| 5  | di_nhon_got         | 11                     |                   |
| 6  | cu_dong_dau         | 12, 24, 27, 30, 34     |                   |
| 7  | chay_nhay           | 16, 17, 21, 31         | Chạy nhảy        |
| 8  | nhin_tay            | 33                     | Nhìn bàn tay      |
| 9  | tay_cham_dau        | 36                     | Tay chạm đầu      |
| 10 | vo_nguc             | 41                     | Vỗ ngực           |


## Cấu trúc mô hình

| Layer (type)            | Output Shape       | Param #   | Connected to            |
|-------------------------|--------------------|-----------|--------------------------|
| input_layer (InputLayer)| (None, 32, 56, 56, 3)| 0         | -                        |
| tubelet_embedding       | (None, 196, 128)   | 196,736   | input_layer[0][0]       |
| positional_encoder      | (None, 196, 128)   | 25,088    | tubelet_embedding       |
| layer_normalization     | (None, 196, 128)   | 256       | positional_encoder       |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add                     | (None, 196, 128)   | 0         | multi_head_attention, positional_encoder |
| layer_normalization     | (None, 196, 128)   | 256       | add                     |
| sequential              | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_1                   | (None, 196, 128)   | 0         | sequential, add         |
| layer_normalization     | (None, 196, 128)   | 256       | add_1                   |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_2                   | (None, 196, 128)   | 0         | multi_head_attention, add_1 |
| layer_normalization     | (None, 196, 128)   | 256       | add_2                   |
| sequential_1            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_3                   | (None, 196, 128)   | 0         | sequential_1, add_2     |
| layer_normalization     | (None, 196, 128)   | 256       | add_3                   |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_4                   | (None, 196, 128)   | 0         | multi_head_attention, add_3 |
| layer_normalization     | (None, 196, 128)   | 256       | add_4                   |
| sequential_2            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_5                   | (None, 196, 128)   | 0         | sequential_2, add_4     |
| layer_normalization     | (None, 196, 128)   | 256       | add_5                   |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_6                   | (None, 196, 128)   | 0         | multi_head_attention, add_5 |
| layer_normalization     | (None, 196, 128)   | 256       | add_6                   |
| sequential_3            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_7                   | (None, 196, 128)   | 0         | sequential_3, add_6     |
| layer_normalization     | (None, 196, 128)   | 256       | add_7                   |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_8                   | (None, 196, 128)   | 0         | multi_head_attention, add_7 |
| layer_normalization     | (None, 196, 128)   | 256       | add_8                   |
| sequential_4            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_9                   | (None, 196, 128)   | 0         | sequential_4, add_8     |
| layer_normalization     | (None, 196, 128)   | 256       | add_9                   |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_10                  | (None, 196, 128)   | 0         | multi_head_attention, add_9 |
| layer_normalization     | (None, 196, 128)   | 256       | add_10                  |
| sequential_5            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_11                  | (None, 196, 128)   | 0         | sequential_5, add_10    |
| layer_normalization     | (None, 196, 128)   | 256       | add_11                  |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_12                  | (None, 196, 128)   | 0         | multi_head_attention, add_11 |
| layer_normalization     | (None, 196, 128)   | 256       | add_12                  |
| sequential_6            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_13                  | (None, 196, 128)   | 0         | sequential_6, add_12    |
| layer_normalization     | (None, 196, 128)   | 256       | add_13                  |
| multi_head_attention    | (None, 196, 128)   | 66,048    | layer_normalization     |
| add_14                  | (None, 196, 128)   | 0         | multi_head_attention, add_13 |
| layer_normalization     | (None, 196, 128)   | 256       | add_14                  |
| sequential_7            | (None, 196, 128)   | 131,712   | layer_normalization     |
| add_15                  | (None, 196, 128)   | 0         | sequential_7, add_14    |
| layer_normalization     | (None, 196, 128)   | 256       | add_15                  |
| global_avg_pooling3d    | (None, 128)        | 0         | layer_normalization     |
| dense_16 (Dense)        | (None, 43)         | 5,547     | global_avg_pooling3d    |

## Tổng số tham số (Parameters)

- **Total params**: `5,441,411` (≈ 20.76 MB)  
- **Trainable params**: `1,813,803` (≈ 6.92 MB)  
- **Non-trainable params**: `0`  
- **Optimizer params**: `3,627,608` (≈ 13.84 MB)

## Tài liệu tham khảo
ViViT: [A Video Vision Transformer](https://arxiv.org/pdf/2103.15691)  
ViViT Model Keras: [A Transformer-based architecture for video classification.](https://github.com/keras-team/keras-io/blob/master/examples/vision/vivit.py)  


