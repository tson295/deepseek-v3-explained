# BÁO CÁO PHÂN TÍCH BÀI BÁO: DeepSeek-V3 Technical Report

> **Nguồn:** DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437v2 (12/2024).
> **Tóm tắt một dòng:** Một mô hình ngôn ngữ Mixture-of-Experts (MoE) **671 tỉ tham số** (chỉ kích hoạt **37 tỉ** mỗi token), đạt hiệu năng ngang GPT-4o / Claude-3.5-Sonnet nhưng **chi phí huấn luyện chỉ ~5,576 triệu USD** — rẻ hơn nhiều bậc so với các mô hình cùng đẳng cấp.

---

## MỤC LỤC

1. [Bối cảnh & vấn đề trọng tâm](#1-bối-cảnh--vấn-đề-trọng-tâm)
2. [Tổng quan đóng góp](#2-tổng-quan-đóng-góp)
3. [Kiến trúc mô hình](#3-kiến-trúc-mô-hình)
   - 3.1 [MLA – Multi-head Latent Attention](#31-mla--multi-head-latent-attention)
   - 3.2 [DeepSeekMoE + Auxiliary-Loss-Free Load Balancing](#32-deepseekmoe--cân-bằng-tải-không-cần-hàm-phụ)
   - 3.3 [MTP – Multi-Token Prediction](#33-mtp--multi-token-prediction)
4. [Hạ tầng & tối ưu huấn luyện](#4-hạ-tầng--tối-ưu-huấn-luyện)
   - 4.1 [DualPipe](#41-dualpipe--song-song-đường-ống-hai-chiều)
   - 4.2 [Kernel all-to-all xuyên node](#42-kernel-all-to-all-xuyên-node)
   - 4.3 [Tiết kiệm bộ nhớ](#43-tiết-kiệm-bộ-nhớ)
   - 4.4 [Huấn luyện FP8](#44-huấn-luyện-fp8--điểm-đột-phá-kỹ-thuật-lớn-nhất)
   - 4.5 [Triển khai suy luận](#45-triển-khai-suy-luận-inference)
   - 4.6 [Đề xuất thiết kế phần cứng](#46-đề-xuất-thiết-kế-phần-cứng)
5. [Tiền huấn luyện (Pre-Training)](#5-tiền-huấn-luyện-pre-training)
6. [Hậu huấn luyện (Post-Training)](#6-hậu-huấn-luyện-post-training)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm)
8. [Ablation & các phát hiện quan trọng](#8-ablation--các-phát-hiện-quan-trọng)
9. [Hạn chế & hướng tương lai](#9-hạn-chế--hướng-tương-lai)
10. [Nhận xét tổng kết](#10-nhận-xét-tổng-kết)

---

## 1. Bối cảnh & vấn đề trọng tâm

Các LLM lớn (GPT-4o, Claude-3.5, Gemini) đang thu hẹp khoảng cách tới AGI, nhưng đa số là **closed-source** và **cực kỳ tốn kém**. Các mô hình open-source (LLaMA, Qwen, Mistral, DeepSeek) cố đuổi theo. Bài toán mà DeepSeek-V3 đặt ra:

> **Làm sao scale một mô hình open-source lên cỡ "frontier" (hàng trăm tỉ tham số) mà chi phí huấn luyện vẫn ở mức chấp nhận được?**

Trả lời của họ là **đồng thiết kế (co-design) thuật toán + framework + phần cứng**: mỗi tầng đều được tối ưu để hỗ trợ tầng kia. Đây là tinh thần xuyên suốt cả bài.

Con số minh chứng (Bảng chi phí trong bài):

| Giai đoạn | GPU-hours (H800) | Chi phí (USD, giả định $2/GPU-giờ) |
|---|---|---|
| Pre-Training | 2.664M | $5.328M |
| Context Extension | 119K | $0.238M |
| Post-Training | 5K | $0.01M |
| **Tổng** | **2.788M** | **$5.576M** |

Mỗi nghìn tỉ (1T) token chỉ tốn 180K GPU-hours ≈ 3,7 ngày trên cụm 2048 GPU. Toàn bộ pre-training (14,8T token) hoàn thành trong **chưa đầy 2 tháng**.

> ⚠️ **Lưu ý quan trọng để không hiểu sai:** Con số $5,576M **chỉ** là chi phí của lần huấn luyện chính thức cuối cùng. Nó **không** bao gồm chi phí nghiên cứu, thử nghiệm kiến trúc, ablation, hay dữ liệu. Đây là điểm hay bị truyền thông phóng đại.

---

## 2. Tổng quan đóng góp

Bài báo có **4 trụ cột đóng góp**, t sẽ giải thích sâu từng cái ở các phần sau:

| # | Trụ cột | Đóng góp cụ thể | Mới ở đâu |
|---|---|---|---|
| 1 | **Kiến trúc** | Auxiliary-loss-free load balancing + Multi-Token Prediction (MTP) | Loại bỏ hàm phụ làm hại hiệu năng; predict nhiều token |
| 2 | **Hiệu quả huấn luyện** | FP8 mixed-precision ở quy mô siêu lớn + DualPipe + kernel all-to-all | **Lần đầu** validate FP8 trên mô hình cực lớn; gần như zero overhead giao tiếp |
| 3 | **Hậu huấn luyện** | Chưng cất (distill) năng lực suy luận từ DeepSeek-R1 | Đưa "long Chain-of-Thought" vào mô hình thường mà vẫn kiểm soát độ dài |
| 4 | **Kết quả** | SOTA open-source, ngang closed-source, chi phí thấp | Mô hình open-source mạnh nhất tại thời điểm phát hành |

Hai kiến trúc nền tảng (MLA và DeepSeekMoE) **không phải là mới** trong bài này — chúng được kế thừa và đã được kiểm chứng ở DeepSeek-V2. Cái **mới** nằm ở 4 trụ cột trên. T vẫn sẽ giải thích MLA/MoE đầy đủ vì chúng là nền tảng để hiểu phần mới.

---

## 3. Kiến trúc mô hình

Khung tổng thể vẫn là **Transformer**. Hai thành phần lõi: **MLA** cho attention, **DeepSeekMoE** cho FFN.

Cấu hình tổng: 61 lớp Transformer, hidden dim 7168, 128 attention heads. 3 lớp đầu là FFN dày (dense), 58 lớp còn lại là MoE. Mỗi lớp MoE có **1 shared expert + 256 routed experts**, mỗi token kích hoạt **8 routed experts**. ⇒ 671B tổng, 37B kích hoạt/token.

### 3.1 MLA – Multi-head Latent Attention

#### Cái cũ làm gì & nhược điểm

Trong suy luận tự hồi quy (autoregressive), mô hình phải lưu **KV cache** — toàn bộ key & value của các token đã sinh — để không phải tính lại. Vấn đề:

- **Multi-Head Attention (MHA)** chuẩn: lưu key + value cho **mọi head** ở mọi token ⇒ KV cache **khổng lồ**. Với context dài (128K token) và mô hình lớn, KV cache ngốn bộ nhớ GPU kinh khủng, trở thành nút thắt cổ chai chính của suy luận.
- Các giải pháp trước đó như **MQA (Multi-Query Attention)** và **GQA (Grouped-Query Attention)** giảm KV cache bằng cách cho nhiều head **chia sẻ** chung key/value. Nhược điểm: **đánh đổi chất lượng** — chia sẻ càng nhiều thì càng mất biểu đạt, hiệu năng giảm.

#### MLA mới gì

MLA dùng **nén liên hợp hạng thấp (low-rank joint compression)** cho key và value. Thay vì lưu key/value đầy đủ, nó chỉ lưu một **vector tiềm ẩn nén** $\mathbf{c}_t^{KV}$ có chiều $d_c \ll d_h n_h$:

```
c_KV = W_DKV · h_t          (nén xuống chiều nhỏ d_c)
k_C  = W_UK · c_KV          (giải nén lại khi cần)
v_C  = W_UV · c_KV
```

Khi suy luận, **chỉ cần cache vector nén $\mathbf{c}_t^{KV}$ và một key mang RoPE tách rời $\mathbf{k}_t^R$** — chứ không phải toàn bộ key/value. Cấu hình thực tế: $d_c = 512$ (so với $d_h \cdot n_h = 128 \times 128 = 16384$) ⇒ KV cache giảm cực mạnh.

Một chi tiết kỹ thuật tinh tế: **RoPE (positional embedding) được tách riêng** ra một nhánh nhỏ ($\mathbf{k}_t^R$, $d_h^R = 64$). Lý do: RoPE phụ thuộc vị trí, không tương thích trực tiếp với phép nén hạng thấp (nếu nén chung sẽ không tách được phép quay theo vị trí ra khỏi ma trận up-projection). Nên họ tách "decoupled key/query" mang RoPE riêng.

Query cũng được nén hạng thấp ($\mathbf{c}_t^Q$, $d_c' = 1536$) — nhưng mục đích khác: **giảm bộ nhớ activation khi huấn luyện**, không phải KV cache.

#### Tại sao đột phá

> MLA đạt **chất lượng ngang MHA đầy đủ** nhưng **KV cache nhỏ như MQA/GQA**. Nó phá vỡ thế đánh đổi "bộ nhớ ↔ chất lượng" mà MQA/GQA mắc phải. Đây là chìa khóa giúp DeepSeek-V3 phục vụ context 128K với chi phí suy luận chấp nhận được.

---

### 3.2 DeepSeekMoE + Cân bằng tải không cần hàm phụ

#### MoE là gì & DeepSeekMoE khác gì MoE truyền thống

**MoE (Mixture-of-Experts):** thay một FFN lớn bằng nhiều "expert" (FFN nhỏ), mỗi token chỉ đi qua một vài expert được **router** chọn. ⇒ tổng tham số khổng lồ nhưng tính toán mỗi token thì ít (sparse activation). Đây là cách "scale tham số mà không scale chi phí tính toán tuyến tính".

**DeepSeekMoE** (kế thừa từ V2) khác MoE truyền thống (GShard) ở hai điểm:
- **Expert chi tiết hơn (finer-grained):** chia thành nhiều expert nhỏ hơn ⇒ tổ hợp linh hoạt hơn, chuyên môn hóa tốt hơn.
- **Shared experts:** một số expert luôn được kích hoạt cho mọi token (học kiến thức chung), tách khỏi routed experts (học kiến thức chuyên biệt).

Điểm khác nhỏ so với V2: V3 dùng hàm **sigmoid** để tính điểm tương hợp token–expert (thay vì softmax), rồi chuẩn hóa trên các expert được chọn.

#### Cái cũ (auxiliary loss) làm gì & tại sao bất ổn

Vấn đề kinh điển của MoE: **routing collapse** (sụp đổ định tuyến). Router có xu hướng **dồn token vào một số ít expert** ưa thích, các expert khác "chết đói". Hậu quả:
- Mất cân bằng tải ⇒ trong môi trường **expert parallelism** (mỗi expert nằm trên GPU khác nhau), GPU chứa expert quá tải bị nghẽn, GPU chứa expert rảnh thì ngồi chơi ⇒ **lãng phí tính toán**.

**Giải pháp truyền thống — auxiliary loss (hàm mất mát phụ):** thêm một số hạng phạt vào loss để **ép** router phân phối đều token. Nhưng đây chính là chỗ **bất ổn / không tối ưu**:

> ⚠️ **Nhược điểm cốt lõi:** Hàm phụ tạo ra **xung đột gradient** với mục tiêu ngôn ngữ chính. Nếu đặt trọng số hàm phụ **quá lớn** → ép cân bằng quá mạnh → **làm hỏng hiệu năng mô hình** (router buộc phải chọn expert không tối ưu cho token đó chỉ để cho "đều"). Nếu đặt **quá nhỏ** → không đủ sức chống collapse. Việc dò trọng số này là một sự đánh đổi mong manh, luôn phải hi sinh một phần chất lượng để đổi lấy cân bằng.

#### Cái mới: Auxiliary-Loss-Free (cân bằng không cần hàm phụ)

Ý tưởng cực kỳ thanh lịch: **tách rời việc cân bằng tải ra khỏi gradient của loss.**

Họ thêm một **bias term $b_i$** cho mỗi expert, **chỉ cộng vào điểm khi quyết định top-K routing**:

```
chọn top-K dựa trên   (s_i,t + b_i)
nhưng gating value vẫn dùng   s_i,t  gốc (KHÔNG có bias)
```

Cơ chế điều chỉnh bias **không qua gradient** mà qua quan sát tải trực tiếp:
- Sau mỗi training step, theo dõi tải thực tế của từng expert trên cả batch.
- Expert nào **quá tải** → **giảm** $b_i$ đi $\gamma$.
- Expert nào **thiếu tải** → **tăng** $b_i$ lên $\gamma$.

($\gamma$ = "bias update speed", đặt 0.001 cho 14,3T token đầu, rồi về 0 cho 500B token cuối.)

#### Tại sao đột phá

> Vì bias **chỉ ảnh hưởng việc *chọn* expert, không ảnh hưởng giá trị gating nhân vào output**, nên nó **không tạo gradient xung đột** với mục tiêu ngôn ngữ. Mô hình được cân bằng tải mà **không phải hi sinh chất lượng**. Đây chính là cái mà auxiliary loss không bao giờ làm được — nó vốn dĩ phải đánh đổi. Ablation (Bảng 4 trong bài) cho thấy phương pháp này thắng auxiliary-loss trên gần như mọi benchmark.

**Bổ trợ:** Vẫn giữ một **sequence-wise balance loss** với trọng số $\alpha$ **cực nhỏ** (0.0001) — chỉ để phòng mất cân bằng cực đoan trong một chuỗi đơn lẻ, không đủ mạnh để làm hại hiệu năng.

**Node-Limited Routing:** mỗi token chỉ được gửi tới tối đa $M=4$ node ⇒ giới hạn chi phí giao tiếp xuyên node, gần đạt overlap hoàn toàn tính toán–giao tiếp.

**No Token-Dropping:** Nhờ cân bằng tốt, V3 **không drop token nào** cả khi huấn luyện lẫn suy luận (nhiều MoE khác phải drop token khi expert quá tải).

#### Phát hiện sâu (Section 4 + Phụ lục B)

Ablation thêm cho thấy: bản chất ưu thế đến từ việc cân bằng **theo batch** (batch-wise) thay vì **theo từng chuỗi** (sequence-wise). Cân bằng theo batch **lỏng hơn**, cho phép expert **chuyên môn hóa theo domain** (vì không ép mỗi chuỗi phải dùng đều mọi expert). Hình minh họa expert load cho thấy mô hình aux-loss-free có **mức chuyên môn hóa expert cao hơn rõ rệt**. Khi họ thử một "batch-wise auxiliary loss" thì nó cũng đạt hiệu năng tương đương aux-loss-free ⇒ xác nhận giả thuyết: **phạm vi cân bằng (batch vs sequence) mới là yếu tố quyết định, không phải bản thân việc có hàm phụ hay không.**

---

### 3.3 MTP – Multi-Token Prediction

#### Cái cũ & động lực

LLM truyền thống huấn luyện bằng **next-token prediction**: mỗi vị trí chỉ dự đoán **1 token kế tiếp**. Tín hiệu huấn luyện khá "thưa" — mỗi token chỉ cho một tín hiệu giám sát.

#### MTP mới gì

MTP mở rộng phạm vi: tại mỗi vị trí, dự đoán **D token tương lai**. Bài này đặt $D=1$ (tức dự đoán thêm 1 token nữa, tổng 2 token).

Khác biệt với công trình trước (Meta MTP dùng **D head song song độc lập**), DeepSeek dùng **D module tuần tự, giữ nguyên chuỗi nhân quả (causal chain) đầy đủ** ở mỗi độ sâu dự đoán:

```
Module thứ k:  h'_i^k = M_k · [ RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k})) ]
               h_i^k  = TRM_k(h'_i^k)          (1 khối Transformer)
               P^k    = OutHead(h_i^k)          (chia sẻ embedding + output head với main model)
```

Mỗi module có 1 khối Transformer riêng + ma trận chiếu $M_k$, nhưng **chia sẻ embedding và output head** với mô hình chính (tiết kiệm bộ nhớ). Loss MTP là cross-entropy trung bình các độ sâu, nhân trọng số $\lambda$ (0.3 cho 10T token đầu, 0.1 sau đó).

#### Hai lợi ích

1. **Làm đặc tín hiệu huấn luyện** ⇒ tăng hiệu quả dữ liệu (mỗi token cho nhiều tín hiệu hơn).
2. **Cho mô hình "lập kế hoạch trước"** biểu diễn để dự đoán xa hơn ⇒ biểu diễn nội tại tốt hơn.

#### Hai cách dùng lúc suy luận

- **Vứt bỏ module MTP** → mô hình chính chạy bình thường (chi phí không đổi). MTP chỉ để **cải thiện chất lượng** lúc train.
- **Hoặc tái sử dụng cho speculative decoding** → tăng tốc sinh. Tỷ lệ chấp nhận token thứ 2 đạt **85–90%**, cho **tốc độ TPS gấp 1,8 lần**.

#### Tại sao đáng chú ý

> MTP là một "free lunch" hiếm có: thêm mục tiêu phụ giúp mô hình chính mạnh hơn (xác nhận qua ablation Bảng 3 — cải thiện trên hầu hết benchmark), mà lúc suy luận **không tốn thêm gì** (vứt module), thậm chí còn **tăng tốc** được nếu muốn (speculative decoding). Việc giữ causal chain đầy đủ giúp mỗi token dự đoán có ngữ cảnh chính xác hơn so với kiểu head song song của Meta.

---

## 4. Hạ tầng & tối ưu huấn luyện

Cụm huấn luyện: **2048 GPU NVIDIA H800**. Mỗi node 8 GPU nối nhau qua **NVLink + NVSwitch**; các node nối nhau qua **InfiniBand (IB)**.

Framework tự xây: **HAI-LLM**. Song song hóa: **16-way Pipeline Parallelism (PP) + 64-way Expert Parallelism (EP) qua 8 node + ZeRO-1 Data Parallelism**. Đáng chú ý: **KHÔNG dùng Tensor Parallelism (TP)** khi train — nhờ tối ưu bộ nhớ tốt.

### 4.1 DualPipe – Song song đường ống hai chiều

#### Cái cũ & nhược điểm

Trong **Pipeline Parallelism**, mô hình bị chia thành nhiều stage trên các GPU khác nhau. Vấn đề kinh điển là **pipeline bubble** (bong bóng) — khoảng thời gian GPU **ngồi chờ** dữ liệu từ stage trước/sau, không làm gì cả ⇒ lãng phí.

- **1F1B** (PipeDream): bubble = $(PP-1)(F+B)$.
- **ZB1P** (ZeroBubble): giảm bubble bằng cách tách backward thành "backward cho input" và "backward cho weights", nhưng vẫn còn bubble.

Thêm nữa, với MoE xuyên node, **giao tiếp all-to-all** (gửi token tới expert ở GPU khác) cực nặng — tỷ lệ tính toán:giao tiếp xấp xỉ **1:1**. Nghĩa là **một nửa thời gian** có thể bị tiêu cho giao tiếp nếu không xử lý khéo.

#### DualPipe mới gì

Ý tưởng cốt lõi: **chồng lấp (overlap) tính toán và giao tiếp** trong một cặp forward + backward chunk. Chia mỗi chunk thành 4 phần: `attention`, `all-to-all dispatch`, `MLP`, `all-to-all combine`. Backward còn tách thêm "backward cho input" và "backward cho weights".

Sau đó **sắp xếp lại** các phần này và **điều chỉnh thủ công tỷ lệ SM** (Streaming Multiprocessor) dành cho giao tiếp vs tính toán, sao cho **giao tiếp được giấu hoàn toàn sau tính toán**. Lịch DualPipe dùng **đường ống hai chiều (bidirectional)** — nạp micro-batch từ **cả hai đầu** đường ống cùng lúc.

Bảng so sánh (trong bài):

| Phương pháp | Bubble | Tham số | Activation |
|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | 1× | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | 1× | $PP$ |
| **DualPipe** | $(\frac{PP}{2}-1)(F\&B+B-3W)$ | **2×** | $PP+1$ |

#### Đánh đổi & tại sao chấp nhận được

DualPipe phải giữ **2 bản sao tham số** (vì hai chiều). Nhưng vì **EP size lớn**, tham số đã được chia nhỏ trên nhiều GPU nên overhead bộ nhớ này không đáng kể. Đổi lại: **bubble giảm mạnh**, và quan trọng nhất —

> **Giao tiếp all-to-all gần như bị giấu hoàn toàn.** Điều này nghĩa là: khi scale mô hình lớn hơn, **chỉ cần giữ tỷ lệ tính toán:giao tiếp cố định**, ta vẫn dùng được fine-grained experts xuyên node với **overhead giao tiếp gần như bằng 0**. Đây là chìa khóa khiến chi phí huấn luyện thấp đến vậy.

### 4.2 Kernel all-to-all xuyên node

Tối ưu ở mức **rất sát phần cứng**, đồng thiết kế với topology mạng:
- **NVLink** (trong node): 160 GB/s ≈ 3,2× băng thông IB (50 GB/s).
- Chiến lược: token đi **IB trước** (tới GPU cùng index ở node đích), rồi **NVLink** chuyển tiếp tới expert đích — IB và NVLink **overlap hoàn toàn**.
- Mỗi token gửi tới tối đa 4 node, trung bình 3,2 expert/node ⇒ có thể scale tới 13 expert mà chi phí giao tiếp không đổi.
- Dùng **warp specialization**: chia 20 SM thành 10 kênh giao tiếp, số warp cho mỗi tác vụ điều chỉnh động.
- Dùng **PTX tùy biến** + auto-tune kích thước chunk ⇒ giảm dùng L2 cache, giảm nhiễu lên SM tính toán khác.

> Kết quả: **chỉ 20/132 SM** đủ để bão hòa cả băng thông IB lẫn NVLink.

### 4.3 Tiết kiệm bộ nhớ

Ba kỹ thuật:
1. **Recompute RMSNorm & MLA up-projection** lúc backward (không lưu activation, tính lại — đổi chút compute lấy nhiều bộ nhớ).
2. **EMA tham số lưu trên CPU**, cập nhật bất đồng bộ ⇒ ước lượng sớm hiệu năng sau learning-rate decay mà không tốn bộ nhớ/thời gian GPU.
3. **Chia sẻ vật lý embedding + output head** giữa module MTP và main model (nhờ đặt lớp nông nhất và sâu nhất trên cùng PP rank).

### 4.4 Huấn luyện FP8 – ĐIỂM ĐỘT PHÁ KỸ THUẬT LỚN NHẤT

> **Đây là phần t khuyên đọc kỹ nhất**, vì nó là đóng góp được nhấn mạnh "lần đầu tiên validate FP8 trên mô hình cực lớn".

#### Cái cũ & tại sao FP8 trước đây bất ổn

Huấn luyện thường dùng **BF16** hoặc **FP16** (16-bit). FP8 (8-bit) chỉ có 8 bit ⇒ **nhanh gấp đôi, tốn nửa bộ nhớ**. Nhưng FP8 cực kỳ khó dùng cho **pre-training quy mô lớn** vì:

1. **Dải động (dynamic range) hẹp** do ít bit mũ ⇒ dễ **overflow/underflow**.
2. **Outliers (giá trị ngoại lai)** trong activation/weight/gradient: chỉ một vài giá trị cực lớn sẽ "kéo căng" thang đo, khiến các giá trị nhỏ bị làm tròn về 0 ⇒ **mất thông tin nghiêm trọng**.
3. Cách chuẩn cũ: scale **toàn tensor** (per-tensor) theo giá trị tuyệt đối lớn nhất ⇒ **cực nhạy với outlier**: một outlier làm hỏng độ chính xác cả tensor.
4. Trước bài này, **rất ít** nghiên cứu chứng minh được FP8 dùng được cho pre-training LLM quy mô lớn — chủ yếu chỉ thành công ở **inference quantization** (vốn dễ hơn nhiều).

#### Cái mới của DeepSeek (4 kỹ thuật chính)

**(a) Fine-Grained Quantization (lượng tử hóa chi tiết)** — *trái tim của giải pháp*

Thay vì scale cả tensor, họ scale theo **nhóm nhỏ**:
- **Activation:** nhóm `1×128` (mỗi token, mỗi 128 kênh) — tile-wise.
- **Weight:** nhóm `128×128` (mỗi 128 input × 128 output channel) — block-wise.

> **Tại sao đột phá:** Outlier giờ chỉ ảnh hưởng **nhóm nhỏ của nó**, không "lây" ra cả tensor. Mỗi nhóm có scale factor riêng, thích ứng cục bộ ⇒ vừa giữ được giá trị lớn vừa không làm chết giá trị nhỏ. Ý tưởng này nhất quán với "microscaling formats" mà GPU thế hệ mới (Blackwell) sau này mới hỗ trợ phần cứng — tức DeepSeek đi trước cả phần cứng.

**(b) Increasing Accumulation Precision (tăng độ chính xác tích lũy)**

Phát hiện một **lỗ hổng phần cứng**: Tensor Core của H800 khi cộng dồn (accumulate) FP8 GEMM **chỉ giữ ~14 bit**, thấp hơn nhiều so với FP32. Khi chiều trong `K` lớn (điển hình ở mô hình lớn), sai số tích lũy lên tới **~2%** — đủ để phá hỏng huấn luyện.

> Giải pháp: **"Promotion to CUDA Cores"** — cứ sau mỗi $N_C = 128$ phần tử, copy kết quả tích lũy trung gian từ Tensor Core sang **thanh ghi FP32 trên CUDA Core** để cộng dồn ở độ chính xác đầy đủ. Khéo léo ở chỗ: trong khi một warpgroup làm "promotion" thì warpgroup kia vẫn chạy MMA ⇒ **overlap, không mất hiệu suất Tensor Core**. $N_C=128$ là ngưỡng tối thiểu cải thiện đáng kể độ chính xác mà không tốn kém.

**(c) Mantissa over Exponents — dùng E4M3 cho mọi tensor**

Công trình trước dùng **hybrid**: `E4M3` (4 bit mũ, 3 bit định trị) cho forward, `E5M2` (5 bit mũ, 2 bit định trị — dải rộng hơn nhưng kém chính xác) cho backward. DeepSeek dùng **E4M3 cho TẤT CẢ** ⇒ độ chính xác cao hơn. Làm được điều này **nhờ fine-grained quantization**: vì chia nhóm nhỏ, các phần tử trong nhóm **chia sẻ bit mũ**, bù đắp cho dải động hẹp của E4M3.

**(d) Online Quantization**

Cách cũ ("delayed quantization") lưu **lịch sử** max-abs của các vòng trước để suy ra scale hiện tại ⇒ phức tạp, kém chính xác. DeepSeek tính max-abs **trực tiếp (online)** cho mỗi tile/block ngay tại chỗ ⇒ scale chính xác hơn, framework đơn giản hơn.

#### Khung mixed-precision

Không phải mọi thứ đều FP8. **GEMM nặng** (Fprop, Dgrad, Wgrad) chạy FP8 (nhanh gấp đôi). Nhưng các thành phần **nhạy cảm** vẫn giữ BF16/FP32: **embedding, output head, MoE gating, normalization, attention**. Master weights, weight gradients, optimizer states giữ độ chính xác cao (nhưng được shard qua nhiều DP rank để giảm overhead). Optimizer states (moment AdamW) dùng **BF16**; master weights + gradients giữ **FP32**.

Activation/communication cũng nén FP8 để giảm bộ nhớ và băng thông MoE (dispatch FP8), nhưng `combine` giữ BF16 ở các điểm quan trọng.

#### Kết quả & bằng chứng

> Sai số loss tương đối của FP8 so với BF16 **luôn dưới 0,25%** — nằm trong khoảng nhiễu ngẫu nhiên của huấn luyện (validated trên 2 mô hình ~16B và ~230B, train ~1T token). Đây là **lần đầu** FP8 được chứng minh khả thi ở quy mô siêu lớn.

#### Bài học về bất ổn định (Phụ lục A.2 — rất quan trọng)

Họ **thử** một cách đơn giản hơn: lượng tử hóa **block-wise `128×128`** cho cả activation gradient (giống cách làm với weight, để chỉ cần transpose lúc backward). **Kết quả: mô hình PHÂN KỲ (diverge)** trên mô hình ~16B sau ~300B token.

> **Nguyên nhân:** Phép `Dgrad` (tính gradient activation, lan ngược về các lớp nông theo chuỗi) **cực kỳ nhạy với độ chính xác**. Activation gradient có **outlier tương quan theo token** (token-correlated outliers) — rất mất cân bằng giữa các token. Block-wise quá thô không xử lý nổi loại outlier này. ⇒ Đây chính là lý do họ **bắt buộc** phải dùng tile-wise `1×128` cho activation (chi tiết hơn, đắt hơn nhưng ổn định). Đoạn này cho thấy ranh giới giữa "thành công" và "phân kỳ" của FP8 mỏng manh thế nào, và vì sao các kỹ thuật trên là **bắt buộc** chứ không phải tùy chọn.

### 4.5 Triển khai suy luận (Inference)

Tách riêng hai giai đoạn với cấu hình khác nhau (vì đặc tính tải khác nhau):

**Prefilling** (xử lý prompt đầu vào):
- Đơn vị tối thiểu: 4 node, 32 GPU. Attention: TP4 + SP + DP8. MoE: EP32.
- **Redundant experts (expert dư thừa):** nhân bản các expert tải cao, phát hiện qua thống kê online, điều chỉnh định kỳ (~10 phút). Mỗi GPU ngoài 8 expert gốc còn host 1 expert dư.
- Xử lý đồng thời 2 micro-batch để giấu overhead giao tiếp.

**Decoding** (sinh từng token):
- Đơn vị tối thiểu: 40 node, 320 GPU. Attention: TP4 + SP + DP80. MoE: EP320 (mỗi GPU 1 expert).
- Shared expert được coi như routed expert "tải nặng luôn được chọn" ⇒ mỗi token chọn 9 expert.
- Dùng giao tiếp **point-to-point qua IB** + công nghệ **IBGDA** để giảm độ trễ.
- Vì decoding nghẽn ở **memory access** (không phải compute), chỉ cần ít SM cho MoE, ưu tiên SM cho attention.

### 4.6 Đề xuất thiết kế phần cứng

Một phần đáng giá: DeepSeek viết hẳn **"thư ngỏ" gửi NVIDIA và các hãng chip**, chỉ ra các thiếu sót phần cứng họ phải "lách":

- **Communication co-processor:** đừng bắt SM (đơn vị tính toán quý giá) làm giao tiếp — hãy có co-processor riêng (như NVIDIA SHARP); hợp nhất IB + NVLink dưới một giao diện.
- **Độ chính xác tích lũy FP8 cao hơn trong Tensor Core** (vấn đề 14-bit ở trên — họ đã phải lách bằng promotion CUDA Core).
- **Hỗ trợ native tile/block-wise quantization** (hiện chỉ có per-tensor).
- **Hỗ trợ online quantization** (fuse FP8 cast với TMA access, near-memory computing).
- **Hỗ trợ transposed GEMM** (tránh phải đọc–giải lượng tử–transpose–lượng tử lại lúc backward).

> Điều này nói lên: V3 bị **giới hạn bởi phần cứng hiện tại**, và các thủ thuật của họ một phần là để **vá** các thiếu sót đó. Nhiều đề xuất sau này được hiện thực hóa trên kiến trúc Blackwell.

---

## 5. Tiền huấn luyện (Pre-Training)

### Dữ liệu
- **14,8T token** chất lượng cao, đa dạng. Tăng tỷ lệ **toán + lập trình**, mở rộng **đa ngôn ngữ** (vượt ra ngoài Anh–Trung).
- **Document packing** (đóng gói tài liệu) nhưng **không** dùng cross-sample attention masking.
- **Fill-in-Middle (FIM)** theo khung Prefix-Suffix-Middle (PSM), tỷ lệ 0,1 — giúp mô hình điền giữa văn bản dựa trên ngữ cảnh mà không hại next-token prediction.
- **Tokenizer:** Byte-level BPE, vocab **128K**. Token mới gộp dấu câu + xuống dòng để nén tốt hơn, nhưng gây **token boundary bias** ⇒ họ ngẫu nhiên tách một phần token gộp lúc train để giảm bias.

### Siêu tham số huấn luyện
- AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1). Max seq length lúc pre-train: **4K**.
- **Learning rate schedule** đặc biệt: warmup tuyến tính 0→$2.2\times10^{-4}$ (2K step) → giữ hằng số tới 10T token → cosine decay xuống $2.2\times10^{-5}$ trong 4,3T token → 2 mức hằng số nhỏ ở 500B token cuối.
- **Batch size scheduling:** tăng dần 3072 → 15360 trong 469B token đầu, rồi giữ 15360.

### Độ ổn định
> Toàn bộ quá trình pre-training **không gặp loss spike không hồi phục nào**, **không phải roll-back lần nào**. Đây là tuyên bố đáng nể với mô hình 671B (các mô hình lớn thường xuyên bị spike/diverge phải roll back).

### Mở rộng context (Long Context Extension)
Dùng **YaRN** mở rộng 2 giai đoạn, mỗi giai đoạn 1000 step: **4K → 32K → 128K**. Kết quả NIAH ("Needle In A Haystack") tốt trên toàn dải tới 128K.

---

## 6. Hậu huấn luyện (Post-Training)

### 6.1 Supervised Fine-Tuning (SFT)
1,5 triệu mẫu, mỗi domain có cách tạo dữ liệu riêng. Fine-tune 2 epoch, cosine LR $5\times10^{-6}$→$1\times10^{-6}$, sample masking khi pack.

**Dữ liệu suy luận (reasoning)** — phần tinh tế nhất:
- Dùng mô hình nội bộ **DeepSeek-R1** sinh dữ liệu. Vấn đề: R1 **suy nghĩ quá nhiều (overthinking), định dạng kém, quá dài**.
- Mục tiêu: cân bằng **độ chính xác cao của R1** với **sự rõ ràng, ngắn gọn**.
- Cách làm: với mỗi domain (code/toán/reasoning), xây một **expert model** qua pipeline SFT+RL. Sinh **2 loại mẫu**: `<problem, original response>` và `<system prompt, problem, R1 response>`. System prompt được thiết kế để hướng mô hình tạo phản hồi có **cơ chế phản tỉnh (reflection) và kiểm chứng (verification)**.
- Giai đoạn RL: dùng high-temperature sampling, mô hình **học cách tích hợp pattern của R1** ngay cả khi không có system prompt. Cuối cùng dùng **rejection sampling** để chắt lọc dữ liệu SFT chất lượng cao cho mô hình cuối.

**Dữ liệu không suy luận (non-reasoning):** dùng DeepSeek-V2.5 sinh, người chú thích kiểm chứng.

### 6.2 Reinforcement Learning (RL)

**Reward Model (2 loại):**
- **Rule-based RM:** với bài có đáp án xác định (toán bắt buộc đóng khung kết quả, code chạy qua compiler/test case). Ưu điểm: **chống reward hacking** (không thể gian lận).
- **Model-based RM:** với bài đáp án tự do. RM huấn luyện từ checkpoint SFT của V3. Để tăng độ tin cậy, dữ liệu preference **kèm cả chain-of-thought dẫn tới reward** ⇒ giảm reward hacking.

**GRPO – Group Relative Policy Optimization:**

> **Cái cũ (PPO):** cần một **critic model** (mô hình giá trị) thường **cùng kích thước với policy model** ⇒ tốn gấp đôi bộ nhớ/tính toán.

> **GRPO mới:** **bỏ hẳn critic model.** Thay vào đó, với mỗi câu hỏi, sample một **nhóm G output**, dùng **điểm trung bình của nhóm làm baseline**. Advantage $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ (chuẩn hóa trong nhóm). ⇒ **Tiết kiệm tài nguyên lớn** mà vẫn ổn định. Có thêm số hạng KL ràng buộc với reference model để không trôi quá xa.

---

## 7. Kết quả thực nghiệm

### 7.1 Base model (so với open-source khác)
DeepSeek-V3-Base **vượt toàn diện** DeepSeek-V2-Base và Qwen2.5-72B-Base, **vượt LLaMA-3.1-405B-Base** trên đa số benchmark ⇒ **mô hình base open-source mạnh nhất**, đặc biệt ở **code và toán**. Ấn tượng: thắng Qwen2.5-72B dù chỉ kích hoạt **một nửa** tham số; thắng LLaMA-3.1-405B dù 405B có **gấp 11 lần** tham số kích hoạt.

### 7.2 Chat model (so với cả closed-source) — Bảng chính

| Nhóm | Benchmark | DeepSeek-V3 | GPT-4o-0513 | Claude-3.5-Sonnet-1022 | Qwen2.5-72B | LLaMA-3.1-405B |
|---|---|---|---|---|---|---|
| English | MMLU | **88.5** | 87.2 | 88.3 | 85.3 | 88.6 |
| | MMLU-Redux | **89.1** | 88.0 | 88.9 | 85.6 | 86.2 |
| | MMLU-Pro | 75.9 | 72.6 | **78.0** | 71.6 | 73.3 |
| | DROP (F1) | **91.6** | 83.7 | 88.3 | 76.7 | 88.7 |
| | GPQA-Diamond | 59.1 | 49.9 | **65.0** | 49.0 | 51.1 |
| | SimpleQA | 24.9 | **38.2** | 28.4 | 9.1 | 17.1 |
| | LongBench v2 | **48.7** | 48.1 | 41.0 | 39.4 | 36.1 |
| Code | HumanEval-Mul | **82.6** | 80.5 | 81.7 | 77.3 | 77.2 |
| | LiveCodeBench (CoT) | **40.5** | 33.4 | 36.3 | 31.1 | 28.4 |
| | Codeforces (%ile) | **51.6** | 23.6 | 20.3 | 24.8 | 25.3 |
| | SWE-Bench Verified | 42.0 | 38.8 | **50.8** | 23.8 | 24.5 |
| | Aider-Polyglot | **49.6** | 16.0 | 45.3 | 7.6 | 5.8 |
| Math | AIME 2024 | **39.2** | 9.3 | 16.0 | 23.3 | 23.3 |
| | MATH-500 | **90.2** | 74.6 | 78.3 | 80.0 | 73.8 |
| | CNMO 2024 | **43.2** | 10.8 | 13.1 | 15.9 | 6.8 |
| Chinese | C-Eval | **86.5** | 76.0 | 76.7 | 86.1 | 61.5 |
| | C-SimpleQA | **64.8** | 59.3 | 51.3 | 48.4 | 50.4 |

**Đọc bảng:**
- **Toán: vượt trội tuyệt đối.** MATH-500 đạt 90.2 (vượt #2 ~10 điểm), AIME/CNMO bỏ xa mọi đối thủ — nhờ distill từ R1. Thậm chí vượt cả o1-preview ở một số benchmark.
- **Code thuật toán (LiveCodeBench, Codeforces): dẫn đầu.** Code kỹ thuật (SWE-Bench): thua Claude-3.5 nhưng vượt mọi open-source.
- **Kiến thức (MMLU/MMLU-Redux): ngang top.**
- **Điểm yếu rõ:** **SimpleQA tiếng Anh (24.9) thua GPT-4o (38.2)** — do V3 phân bổ nhiều token học **kiến thức tiếng Trung** hơn (đổi lại C-SimpleQA dẫn đầu 64.8). GPQA thua Claude. SWE-Bench thua Claude.

### 7.3 Open-ended (LLM làm giám khảo)

| Model | Arena-Hard | AlpacaEval 2.0 |
|---|---|---|
| **DeepSeek-V3** | **85.5** | **70.0** |
| Claude-3.5-Sonnet-1022 | 85.2 | 52.0 |
| GPT-4o-0513 | 80.4 | 51.1 |

> V3 là **mô hình open-source đầu tiên vượt 85% trên Arena-Hard**. AlpacaEval 2.0 đạt 70.0 — bỏ xa mọi đối thủ (cả closed-source).

### 7.4 V3 làm Generative Reward Model
Trên RewardBench, V3 (87.0) ngang GPT-4o-0806 (86.7) và Claude-3.5-1022 (88.7); với voting maj@6 đạt **89.6** (cao nhất). ⇒ V3 đủ tốt để **tự làm giám khảo** cho chính quá trình alignment của nó (self-rewarding).

---

## 8. Ablation & các phát hiện quan trọng

| Ablation | Phát hiện |
|---|---|
| **MTP** (Bảng 3) | Cải thiện hiệu năng trên hầu hết benchmark ở cả 2 quy mô (15.7B & 228.7B). HumanEval tăng mạnh (44.5→53.7 ở quy mô lớn). |
| **Aux-Loss-Free** (Bảng 4) | Thắng aux-loss-based trên gần như mọi benchmark. GSM8K 70.7→74.5. |
| **Batch vs Sequence balance** | Batch-wise balance (dù aux-loss-free hay batch-wise aux-loss) đều thắng sequence-wise. Validation loss 1B: 2.258 (seq) vs 2.253 (free/batch). ⇒ phạm vi cân bằng mới là yếu tố quyết định. |
| **FP8 vs BF16** (Phụ lục A.1) | Sai số tương đối < 0.25%, trong ngưỡng nhiễu. |
| **Block-wise quant Dgrad** (Phụ lục A.2) | **Phân kỳ mô hình!** Activation gradient có token-correlated outliers, block-wise không xử lý nổi. ⇒ buộc dùng tile-wise. |
| **R1 Distillation** (Bảng 6) | LiveCodeBench 31.1→37.4, MATH-500 74.6→83.2. Nhưng **độ dài phản hồi tăng mạnh** (MATH: 769→1510 token) ⇒ phải đánh đổi accuracy vs length. |
| **MTP speculative decoding** | Tỷ lệ chấp nhận token thứ 2: 85–90% ⇒ **1.8× TPS**. |

---

## 9. Hạn chế & hướng tương lai

**Hạn chế (tác giả tự nêu):**
1. **Đơn vị triển khai lớn** (prefilling 32 GPU, decoding 320 GPU) ⇒ **gánh nặng cho nhóm nhỏ**. Không thể chạy trên vài GPU.
2. Tốc độ sinh đã gấp 2× V2 nhưng **vẫn còn dư địa cải thiện**.
3. (Ngầm) Phụ thuộc vào các thủ thuật lách thiếu sót phần cứng hiện tại.

**Hướng tương lai:**
- Cải tiến kiến trúc hướng tới **context vô hạn**, vượt giới hạn Transformer.
- Scale dữ liệu đa chiều hơn.
- Mở rộng **deep thinking** (reasoning dài hơn, sâu hơn).
- Phương pháp đánh giá đa chiều, tránh over-fit benchmark cố định.

---

## 10. Nhận xét tổng kết

**Vì sao bài này quan trọng:**

1. **Phá vỡ định kiến "mô hình frontier phải đắt".** Co-design thuật toán–framework–phần cứng cho thấy có thể train mô hình 671B với <6 triệu USD (chi phí lần train chính thức).

2. **Mỗi đóng góp giải quyết một đánh đổi cố hữu:**
   - MLA: phá đánh đổi *bộ nhớ KV ↔ chất lượng*.
   - Aux-loss-free: phá đánh đổi *cân bằng tải ↔ chất lượng*.
   - MTP: thêm tín hiệu train mà *không tốn chi phí suy luận*.
   - FP8: phá đánh đổi *tốc độ ↔ ổn định số học*.
   - DualPipe: phá đánh đổi *song song ↔ overhead giao tiếp*.
   - GRPO: phá đánh đổi *RL chất lượng ↔ chi phí critic*.

3. **Đóng góp ngược cho cộng đồng phần cứng:** phần "Suggestions on Hardware Design" là hiếm thấy — một lab AI chỉ ra cụ thể chip cần gì.

4. **Distillation từ R1** mở ra hướng: đưa năng lực reasoning long-CoT vào mô hình thường — tiền đề cho làn sóng "reasoning models" sau này.

**Điểm cần đọc kỹ nhất nếu làm báo cáo học thuật:** Phần **FP8 (4.4 + Phụ lục A.2)** và **Auxiliary-Loss-Free (3.2)** — đây là hai chỗ thể hiện rõ nhất tư duy "tại sao cái cũ bất ổn và cái mới giải quyết thế nào", và là nơi đóng góp khoa học sắc nét nhất.

**Điểm cần thận trọng khi trích dẫn:** Con số chi phí $5,576M — luôn kèm chú thích "chỉ là lần train chính thức cuối", tránh hiểu lầm.

---

*Báo cáo được tổng hợp từ việc đọc toàn văn main.tex, content/fp8.tex, các bảng và phụ lục của arXiv:2412.19437v2.*
