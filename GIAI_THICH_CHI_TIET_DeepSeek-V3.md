# GIẢI THÍCH CHI TIẾT TỪNG PHẦN — DeepSeek-V3

> File này đào sâu **mọi phần quan trọng** của bài báo, **giải thích từng công thức kèm ví dụ số cụ thể**. Mục tiêu: đọc xong là hiểu *bản chất* chứ không chỉ đọc được ký hiệu.
>
> Đọc kèm file `BAO_CAO_DeepSeek-V3.md` (báo cáo tổng quan) và `FAQ_DeepSeek-V3.md` (hỏi–đáp).

**Các hằng số cấu hình của DeepSeek-V3 dùng xuyên suốt các ví dụ:**

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| $L$ | số lớp Transformer | 61 |
| $d$ | hidden dimension | 7168 |
| $n_h$ | số attention head | 128 |
| $d_h$ | chiều mỗi head | 128 |
| $d_c$ | chiều nén KV | 512 |
| $d_c'$ | chiều nén query | 1536 |
| $d_h^R$ | chiều decoupled key/query (RoPE) | 64 |
| $N_s$ | số shared expert | 1 |
| $N_r$ | số routed expert | 256 |
| $K_r$ | số routed expert kích hoạt/token | 8 |
| — | intermediate dim mỗi expert | 2048 |
| $V$ | vocab size | 128K |
| — | tổng / kích hoạt tham số | 671B / 37B |

---

## MỤC LỤC

1. [Giải mã con số 671B / 37B — MoE hoạt động thế nào](#1-giải-mã-con-số-671b--37b)
2. [MLA — giải thích từng công thức + ví dụ số](#2-mla--giải-thích-từng-công-thức--ví-dụ-số)
3. [DeepSeekMoE gating — từng công thức](#3-deepseekmoe-gating--từng-công-thức)
4. [Auxiliary-Loss-Free — ví dụ cập nhật bias từng bước](#4-auxiliary-loss-free--ví-dụ-cập-nhật-bias)
5. [Sequence-wise balance loss — bóc tách công thức](#5-sequence-wise-balance-loss)
6. [MTP — đường đi của dữ liệu + công thức loss](#6-mtp--đường-đi-dữ-liệu--công-thức-loss)
7. [FP8 — toán học của lượng tử hóa & vì sao bất ổn](#7-fp8--toán-học-của-lượng-tử-hóa--vì-sao-bất-ổn)
8. [DualPipe — công thức bubble + ví dụ số](#8-dualpipe--công-thức-bubble--ví-dụ-số)
9. [All-to-all — toán học băng thông](#9-all-to-all--toán-học-băng-thông)
10. [GRPO — bóc tách hàm mục tiêu + ví dụ nhóm reward](#10-grpo--bóc-tách-hàm-mục-tiêu--ví-dụ-nhóm-reward)
11. [Chi phí huấn luyện — kiểm chứng các con số](#11-chi-phí-huấn-luyện--kiểm-chứng-con-số)
12. [Long context (YaRN) — ý tưởng cốt lõi](#12-long-context-yarn)

---

## 1. Giải mã con số 671B / 37B

Nhiều người nghe "671B tham số nhưng chỉ kích hoạt 37B" mà không hiểu vì sao. Đây là bản chất của **MoE thưa (sparse)**.

### Tham số của một expert

Mỗi expert là một FFN kiểu **SwiGLU**, có 3 ma trận chiếu: `gate`, `up` (đều $d \times d_{ff}$) và `down` ($d_{ff} \times d$), với $d=7168$, $d_{ff}=2048$:

$$\text{params/expert} = 3 \times d \times d_{ff} = 3 \times 7168 \times 2048 \approx 44.0 \text{ triệu}$$

### Tổng tham số (vì sao 671B)

- 58 lớp MoE (3 lớp đầu là dense, không phải MoE).
- Mỗi lớp MoE: 256 routed + 1 shared = 257 expert.

$$\text{params routed/lớp} = 256 \times 44.0\text{M} \approx 11.3 \text{ tỉ}$$
$$\text{tất cả lớp MoE} \approx 11.3\text{B} \times 58 \approx 654 \text{ tỉ}$$

Cộng thêm attention (MLA), embedding, 3 lớp dense, shared experts → **~671 tỉ**. *Phần lớn tham số nằm ở routed experts.*

### Tham số kích hoạt mỗi token (vì sao chỉ 37B)

Mỗi token chỉ đi qua **8 routed + 1 shared = 9 expert** (trên 257):

$$\text{params kích hoạt routed/lớp} = 9 \times 44.0\text{M} \approx 396 \text{ triệu}$$
$$\text{tất cả lớp MoE} \approx 396\text{M} \times 58 \approx 23 \text{ tỉ}$$

Cộng attention + embedding + dense → **~37 tỉ**.

### Ý nghĩa

> Mô hình **"biết" như một mô hình 671B** (vì có ngần ấy tham số để lưu kiến thức), nhưng mỗi token **chỉ "tốn tính toán" như một mô hình ~37B**. Đây là cách tách rời **dung lượng kiến thức** khỏi **chi phí tính toán** — lý do MoE hấp dẫn. Cái giá phải trả: tốn bộ nhớ để **chứa** toàn bộ 671B tham số (dù không tính hết), và độ phức tạp routing/giao tiếp.

---

## 2. MLA — giải thích từng công thức + ví dụ số

### Vấn đề: KV cache

Khi sinh token thứ $t$, attention cần "nhìn lại" key/value của **tất cả** token $1..t-1$. Để khỏi tính lại, ta **cache** chúng. Cache này lớn dần theo context và là **nút nghẽn bộ nhớ chính** khi suy luận.

### Công thức MLA cho Key–Value (giải thích từng dòng)

$$\mathbf{c}_t^{KV} = W^{DKV}\,\mathbf{h}_t \qquad (1)$$

- $\mathbf{h}_t \in \mathbb{R}^{7168}$: đầu vào token $t$.
- $W^{DKV} \in \mathbb{R}^{512 \times 7168}$: ma trận **"down-projection"** (nén xuống).
- **Kết quả $\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$**: vector tiềm ẩn nén. *Đây là thứ DUY NHẤT (cùng $\mathbf{k}_t^R$) cần cache.*

$$\mathbf{k}_t^C = W^{UK}\,\mathbf{c}_t^{KV}, \qquad \mathbf{v}_t^C = W^{UV}\,\mathbf{c}_t^{KV} \qquad (2)(3)$$

- $W^{UK}, W^{UV} \in \mathbb{R}^{16384 \times 512}$: **"up-projection"** (giải nén ngược về $n_h \cdot d_h = 128\times128 = 16384$) — chỉ thực hiện **khi cần dùng**, không cache kết quả.

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR}\,\mathbf{h}_t) \qquad (4)$$

- $\mathbf{k}_t^R \in \mathbb{R}^{64}$: key nhỏ mang thông tin **vị trí (RoPE)**, **tách riêng**. Cũng cần cache.

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C\,;\,\mathbf{k}_t^R] \qquad (5)$$

- Key cuối cùng cho head $i$ = ghép phần nội dung (giải nén) + phần vị trí (chung cho mọi head).

### Tại sao phải TÁCH RoPE ra?

RoPE áp một **phép quay phụ thuộc vị trí** lên key/query. Nếu nhét RoPE vào trong phép nén hạng thấp, ma trận up-projection $W^{UK}$ sẽ bị "vướng" phép quay theo từng vị trí khác nhau ⇒ **không thể gộp $W^{UK}$ vào tính toán trước (absorb)** ⇒ mất hết lợi ích nén. Giải pháp: dành một nhánh nhỏ riêng ($d_h^R=64$) chỉ để mang RoPE, phần nội dung còn lại nén thoải mái.

### VÍ DỤ SỐ: MLA tiết kiệm bao nhiêu?

**Cache mỗi token mỗi lớp:**

| | Cần cache gì | Số phần tử |
|---|---|---|
| **MHA chuẩn** | K đầy đủ ($n_h d_h$) + V đầy đủ ($n_h d_h$) | $2 \times 128 \times 128 = 32768$ |
| **MLA** | $\mathbf{c}_t^{KV}$ (512) + $\mathbf{k}_t^R$ (64) | $512 + 64 = 576$ |

$$\text{Tỷ lệ giảm} = \frac{32768}{576} \approx \mathbf{56.9\times}$$

**Quy ra dung lượng thật** (context 128K token, 61 lớp, lưu BF16 = 2 byte):

- MLA: $576 \times 61 \times 128000 \times 2 \text{ byte} \approx \mathbf{9\ GB}$
- MHA: $32768 \times 61 \times 128000 \times 2 \text{ byte} \approx \mathbf{512\ GB}$ (bất khả thi!)

> **Kết luận:** MLA biến việc phục vụ context 128K từ "bất khả thi" (512 GB chỉ riêng KV cache, vượt xa 1 GPU 80GB) thành "khả thi" (~9 GB). Mà chất lượng vẫn ngang MHA đầy đủ — đó là điểm đột phá.

*(Query cũng nén qua $\mathbf{c}_t^Q \in \mathbb{R}^{1536}$ nhưng mục đích là giảm bộ nhớ activation lúc TRAIN, không liên quan KV cache.)*

---

## 3. DeepSeekMoE gating — từng công thức

Công thức tính output FFN của token $t$:

$$\mathbf{h}_t' = \mathbf{u}_t + \underbrace{\sum_{i=1}^{N_s}\text{FFN}_i^{(s)}(\mathbf{u}_t)}_{\text{shared: LUÔN chạy}} + \underbrace{\sum_{i=1}^{N_r} g_{i,t}\,\text{FFN}_i^{(r)}(\mathbf{u}_t)}_{\text{routed: chỉ chạy 8/256}} \qquad (6)$$

- Số hạng 1: residual (cộng đầu vào).
- Số hạng 2: **shared expert** — luôn kích hoạt cho mọi token (học kiến thức chung).
- Số hạng 3: **routed experts** — chỉ những expert có $g_{i,t} \neq 0$ mới đóng góp.

### Cách tính gating value $g_{i,t}$

$$s_{i,t} = \text{Sigmoid}(\mathbf{u}_t^\top \mathbf{e}_i) \qquad (9)$$

- $\mathbf{e}_i$: vector "centroid" đại diện expert $i$. $s_{i,t}$ = **độ hợp** giữa token $t$ và expert $i$, qua sigmoid ⇒ nằm trong (0,1).

$$g_{i,t}' = \begin{cases} s_{i,t} & \text{nếu } s_{i,t} \in \text{Top-8 trong } \{s_{j,t}\} \\ 0 & \text{ngược lại}\end{cases} \qquad (8)$$

- Chỉ **8 expert có điểm cao nhất** được giữ, còn lại = 0.

$$g_{i,t} = \frac{g_{i,t}'}{\sum_{j=1}^{N_r} g_{j,t}'} \qquad (7)$$

- Chuẩn hóa để tổng trọng số 8 expert = 1.

### VÍ DỤ SỐ (rút gọn còn 6 expert, chọn top-2)

Giả sử điểm hợp: $s = [0.8, 0.1, 0.6, 0.3, 0.7, 0.2]$.
- Top-2 = expert 1 (0.8) và expert 5 (0.7).
- $g_1' = 0.8,\ g_5' = 0.7$, còn lại 0.
- Chuẩn hóa: $g_1 = \frac{0.8}{1.5} = 0.533,\ g_5 = \frac{0.7}{1.5} = 0.467$.
- Output ≈ $\mathbf{u}_t + \text{shared}(\mathbf{u}_t) + 0.533\,\text{FFN}_1(\mathbf{u}_t) + 0.467\,\text{FFN}_5(\mathbf{u}_t)$.

> Lưu ý: V3 dùng **sigmoid** (khác V2 dùng softmax trên toàn bộ). Sigmoid cho điểm độc lập từng expert, tránh cạnh tranh softmax làm điểm bị "nén".

---

## 4. Auxiliary-Loss-Free — ví dụ cập nhật bias

### Vấn đề "routing collapse" bằng số

Giả sử không cân bằng: trong 1 batch 1000 token, expert A nhận 400 token, expert B nhận 5 token. Nếu A,B nằm trên 2 GPU khác nhau (expert parallelism), GPU-A è cổ xử lý 400, GPU-B xử lý 5 token rồi **ngồi chờ** ⇒ lãng phí ~99% GPU-B.

### Cách CŨ (auxiliary loss) và vì sao hại

Thêm $\mathcal{L}_{aux} = \alpha \sum_i f_i P_i$ vào tổng loss. Gradient của nó **ép router** chọn đều hơn — nhưng gradient này **trộn vào** gradient ngôn ngữ, kéo router rời khỏi lựa chọn tối ưu cho dự đoán. $\alpha$ lớn → cân bằng tốt nhưng hại chất lượng; $\alpha$ nhỏ → không đủ chống collapse. **Luôn phải đánh đổi.**

### Cách MỚI: bias term ngoài gradient

$$g_{i,t}' = \begin{cases} s_{i,t} & \text{nếu } (s_{i,t} + b_i) \in \text{Top-K} \{s_{j,t}+b_j\} \\ 0 & \text{ngược lại}\end{cases} \qquad (16)$$

**Mấu chốt:**
- $b_i$ chỉ tham gia việc **CHỌN** expert (so sánh top-K).
- Giá trị nhân vào output vẫn là $s_{i,t}$ **gốc** (không có $b_i$).
- $b_i$ **không** được cập nhật bằng gradient, mà bằng **luật quan sát tải**.

Luật cập nhật sau mỗi step:
$$b_i \leftarrow b_i - \gamma \quad \text{nếu expert } i \text{ QUÁ tải}$$
$$b_i \leftarrow b_i + \gamma \quad \text{nếu expert } i \text{ THIẾU tải}$$

### VÍ DỤ SỐ (3 expert, $\gamma=0.001$, tải lý tưởng = 33% mỗi expert)

**Step 1:** tải đo được A=50%, B=30%, C=20%. Bias đầu $b=[0,0,0]$.
- A quá tải → $b_A = -0.001$
- B ~ cân bằng (coi như hơi thiếu) → $b_B = +0.001$
- C thiếu tải → $b_C = +0.001$

**Step 2:** với token mà $s = [0.50, 0.49, 0.48]$ (A hơi cao nhất):
- Điểm có bias: $s+b = [0.499, 0.491, 0.481]$. A vẫn top-1, nhưng **khoảng cách thu hẹp**.
- Sau vài chục step, $b_A$ giảm đủ để các token "ranh giới" chuyển bớt sang B, C.
- **Quan trọng:** khi A được chọn, output vẫn nhân $s_A = 0.50$ gốc — *router không bị bóp méo giá trị*.

> **Tại sao đột phá:** Việc cân bằng tải được **tách hoàn toàn** khỏi tín hiệu học ngôn ngữ. Không có gradient phụ nào kéo lùi chất lượng. Kết quả ablation: aux-loss-free thắng aux-loss trên gần như mọi benchmark (vd GSM8K 70.7 → 74.5). Cuối training (500B token cuối) họ đặt $\gamma=0$ vì tải đã ổn định.

---

## 5. Sequence-wise balance loss

Dù chủ yếu dựa vào aux-loss-free, vẫn giữ một loss phụ **rất nhẹ** ($\alpha = 0.0001$) để phòng một **chuỗi đơn lẻ** quá lệch:

$$\mathcal{L}_{Bal} = \alpha \sum_{i=1}^{N_r} f_i P_i$$

Bóc tách:
$$f_i = \frac{N_r}{K_r T}\sum_{t=1}^{T}\mathbb{1}(\text{expert } i \in \text{Top-K của token } t)$$

- $f_i$ = **tần suất** expert $i$ được chọn trong chuỗi, đã chuẩn hóa sao cho nếu cân bằng hoàn hảo thì $f_i \approx 1$.

$$P_i = \frac{1}{T}\sum_{t=1}^{T} s_{i,t}', \qquad s_{i,t}' = \frac{s_{i,t}}{\sum_j s_{j,t}}$$

- $P_i$ = **điểm hợp trung bình** (đã chuẩn hóa) của expert $i$ trên chuỗi.

**Trực giác:** $\mathcal{L}_{Bal}$ phạt khi một expert **vừa được chọn nhiều ($f_i$ cao) vừa có điểm cao ($P_i$ cao)** → đẩy phân phối về đều. Vì $\alpha$ cực nhỏ nên nó chỉ là "lưới an toàn", không đủ mạnh để hại hiệu năng.

---

## 6. MTP — đường đi dữ liệu + công thức loss

### Ý tưởng

Mô hình thường: mỗi vị trí dự đoán **1** token kế. MTP: dự đoán thêm **D** token tương lai (V3 dùng $D=1$, tức tổng 2 token).

### Công thức từng bước (module thứ $k$)

$$\mathbf{h}_i'^k = M_k\,[\,\text{RMSNorm}(\mathbf{h}_i^{k-1})\,;\,\text{RMSNorm}(\text{Emb}(t_{i+k}))\,] \qquad (21)$$

- Ghép: biểu diễn token $i$ ở độ sâu trước ($\mathbf{h}_i^{k-1}$, chiều $d$) **+** embedding của token tương lai $t_{i+k}$ (chiều $d$) → vector $2d$.
- $M_k \in \mathbb{R}^{d \times 2d}$ chiếu về lại chiều $d$.
- Khi $k=1$: $\mathbf{h}_i^0$ = biểu diễn từ **mô hình chính**.

$$\mathbf{h}_i^k = \text{TRM}_k(\mathbf{h}_i'^k) \qquad (22)$$

- Đưa qua **1 khối Transformer riêng** của module $k$.

$$P_{i+k+1}^k = \text{OutHead}(\mathbf{h}_i^k) \qquad (23)$$

- Dùng **chung output head** với mô hình chính để ra phân phối xác suất token $t_{i+k+1}$.

### Loss

$$\mathcal{L}_{MTP}^k = -\frac{1}{T}\sum_i \log P_i^k[t_i] \quad\text{(cross-entropy ở độ sâu } k)$$
$$\mathcal{L}_{MTP} = \frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{MTP}^k$$

- $\lambda$ = trọng số: **0.3** cho 10T token đầu, **0.1** cho 4.8T token sau.

### Khác biệt với Meta MTP

Meta dùng **D head song song độc lập** (cùng nhìn một biểu diễn, dự đoán D token riêng rẽ). DeepSeek dùng **D module tuần tự, giữ causal chain đầy đủ**: module $k$ dùng output của module $k-1$. ⇒ mỗi dự đoán có ngữ cảnh nhân quả chính xác hơn.

### Hai chế độ suy luận

1. **Vứt module MTP** → mô hình chính chạy bình thường, chi phí y hệt baseline. (MTP chỉ để cải thiện chất lượng lúc train — xác nhận qua ablation.)
2. **Giữ để speculative decoding** → module MTP "đoán trước" token kế, mô hình chính xác minh. Tỷ lệ chấp nhận 85–90% ⇒ **TPS × 1.8**.

> **Vì sao là "bữa trưa miễn phí":** thêm tín hiệu giám sát dày đặc giúp mô hình chính học tốt hơn, mà lúc deploy có thể bỏ đi (không tốn gì) hoặc dùng để tăng tốc.

---

## 7. FP8 — toán học của lượng tử hóa & vì sao bất ổn

### FP8 là gì

8 bit. Định dạng **E4M3** = 1 bit dấu + 4 bit mũ + 3 bit định trị (mantissa). Dải biểu diễn xấp xỉ **±448**, và chỉ có **3 bit mantissa** ⇒ độ phân giải tương đối thô (~bước nhảy 12.5% giữa các giá trị liền kề trong cùng mức mũ). So sánh: BF16 có 8 bit mantissa, FP32 có 23 bit.

### Vì sao FP8 "ngây thơ" (per-tensor) bất ổn — VÍ DỤ SỐ

Cách cũ: scale **cả tensor** theo max tuyệt đối → max ánh xạ về 448.

Giả sử một tensor có hầu hết giá trị trong $[-2, 2]$ **nhưng có 1 outlier = 1000**:
- Scale = $448 / 1000 = 0.448$.
- Giá trị bình thường $0.5$ → $0.5 \times 0.448 = 0.224$ sau scale.
- Nhưng giá trị nhỏ $0.01$ → $0.01 \times 0.448 = 0.00448$ → **gần dưới ngưỡng biểu diễn của E4M3** ⇒ bị làm tròn về 0 hoặc mất gần hết mantissa.

> **Một outlier 1000 "kéo căng" thang đo, làm chết độ chính xác của TẤT CẢ giá trị nhỏ trong tensor.** Đây chính là nguồn gốc bất ổn.

### Giải pháp 1: Fine-Grained Quantization

Chia tensor thành **nhóm nhỏ**, mỗi nhóm một scale riêng:
- Activation: nhóm `1×128` (per-token, 128 kênh).
- Weight: nhóm `128×128`.

**Cùng ví dụ trên, nhưng chia nhóm:** nếu outlier 1000 chỉ nằm ở nhóm B:
- Nhóm A (max = 2): scale = $448/2 = 224$ → giá trị $0.5$ → $112$, **giữ trọn độ phân giải**.
- Nhóm B (max = 1000): scale = 0.448 → chỉ nhóm B bị thô.
- ⇒ **Outlier bị "cô lập", không lây sang nhóm khác.** 255/256 nhóm vẫn chính xác.

Thêm: vì E4M3 chia sẻ bit mũ trong nhóm nhỏ, dải động hẹp được bù đắp ⇒ dùng được **E4M3 cho mọi tensor** (thay vì hybrid E4M3/E5M2 như cũ).

### Giải pháp 2: Tăng độ chính xác tích lũy — VÍ DỤ SỐ

**Lỗ hổng phần cứng H800:** Tensor Core khi cộng dồn FP8 GEMM **chỉ giữ ~14 bit**. Khi chiều trong $K$ lớn (vd $K=4096$), sai số tích lũy lên tới **~2%** — đủ làm hỏng training.

**Vì sao:** cộng 4096 tích FP8 với chỉ 14 bit accumulator → các bit thấp bị cắt liên tục, lỗi cộng dồn.

**Giải pháp "promotion to CUDA Cores":** cứ sau mỗi $N_C=128$ phần tử, copy tổng trung gian sang **thanh ghi FP32** trên CUDA Core, cộng ở độ chính xác đầy đủ.
- $N_C=128$ = 4 WGMMA — ngưỡng tối thiểu cải thiện rõ rệt mà không tốn kém.
- Khéo: trong khi 1 warpgroup "promotion", warpgroup kia chạy MMA ⇒ **overlap**, Tensor Core không nghỉ.

### Giải pháp 3: Online Quantization

Cách cũ ("delayed"): lưu lịch sử max-abs các vòng trước để đoán scale → trễ, sai. Cách mới: tính max-abs **ngay tại chỗ** cho mỗi tile/block → scale chính xác.

### Bằng chứng bất ổn (Phụ lục A.2 — VÍ DỤ KINH ĐIỂN)

Họ thử lượng tử hóa **block-wise `128×128`** cho **activation gradient** (Dgrad) để đơn giản hóa. **Kết quả: mô hình ~16B PHÂN KỲ sau ~300B token.**

> **Nguyên nhân:** Activation gradient có **token-correlated outliers** — cực mất cân bằng giữa các token. `Dgrad` lan ngược về lớp nông theo chuỗi nên cực nhạy. Block `128×128` quá thô, gộp cả token outlier với token thường vào 1 scale ⇒ phá hỏng gradient ⇒ diverge.
>
> **Bài học:** Đây là lý do họ **bắt buộc** dùng tile `1×128` (chi tiết hơn) cho activation. Ranh giới giữa "FP8 thành công" và "phân kỳ" rất mỏng — các kỹ thuật trên là **điều kiện sống còn**, không phải tùy chọn.

### Kết quả cuối

Sai số loss FP8 vs BF16 **< 0.25%** (trong ngưỡng nhiễu ngẫu nhiên), validated trên mô hình ~16B và ~230B. **Lần đầu** FP8 được chứng minh khả thi ở quy mô siêu lớn. Đổi lại: GEMM nhanh ~2×, bộ nhớ activation giảm mạnh.

---

## 8. DualPipe — công thức bubble + ví dụ số

### Bubble là gì

Trong pipeline parallelism, mô hình chia thành $PP$ stage trên $PP$ GPU. GPU stage sau phải **chờ** stage trước xong mới có việc → thời gian chờ = **bubble** (lãng phí).

### Bảng công thức (trong bài)

| Phương pháp | Bubble |
|---|---|
| 1F1B | $(PP-1)(F+B)$ |
| ZB1P | $(PP-1)(F+B-2W)$ |
| **DualPipe** | $(\frac{PP}{2}-1)(F\&B + B - 3W)$ |

- $F$ = thời gian 1 forward chunk; $B$ = 1 full backward; $W$ = "backward cho weights"; $F\&B$ = 1 cặp forward+backward **đã overlap**.

### VÍ DỤ SỐ MINH HỌA (đặt $PP=16$, $F=1$, $B=2$, $W=1$, $F\&B\approx2$)

- **1F1B:** $(16-1)(1+2) = 15 \times 3 = \mathbf{45}$ đơn vị bubble.
- **ZB1P:** $(16-1)(1+2-2) = 15 \times 1 = \mathbf{15}$.
- **DualPipe:** $(\frac{16}{2}-1)(2+2-3) = 7 \times 1 = \mathbf{7}$.

*(Con số minh họa, không phải đo thực — nhưng cho thấy thứ tự độ lớn: DualPipe bubble ~1/6 so với 1F1B.)*

### Ý tưởng cốt lõi

Chia mỗi chunk thành 4 phần: `attention`, `all-to-all dispatch`, `MLP`, `all-to-all combine`. Sắp xếp lại + điều chỉnh tỷ lệ SM sao cho **giao tiếp được giấu sau tính toán**. Dùng **đường ống 2 chiều** — nạp micro-batch từ cả 2 đầu.

### Đánh đổi

DualPipe giữ **2 bản sao tham số** (vì 2 chiều) → tốn bộ nhớ hơn. Nhưng vì EP size lớn (tham số đã chia nhỏ trên nhiều GPU) nên overhead không đáng kể. **Quan trọng nhất:** giao tiếp all-to-all (vốn ~1:1 với tính toán) **bị giấu gần hết** ⇒ scale lớn hơn mà overhead giao tiếp vẫn ~0.

---

## 9. All-to-all — toán học băng thông

### Bối cảnh

MoE xuyên node phải gửi token tới expert ở GPU/node khác qua mạng. Đây là **giao tiếp all-to-all** — cực nặng.

### Hai loại mạng

| Mạng | Phạm vi | Băng thông |
|---|---|---|
| **NVLink** | trong node (8 GPU) | 160 GB/s |
| **InfiniBand (IB)** | xuyên node | 50 GB/s |

Tỷ lệ: $160 / 50 = \mathbf{3.2\times}$.

### Chiến lược "IB trước, NVLink sau"

1. Token gửi qua **IB** tới GPU **cùng in-node index** ở node đích.
2. Tới node đích, dùng **NVLink** chuyển tiếp tới GPU chứa expert đích.
3. IB và NVLink **overlap hoàn toàn**.

### Toán học giới hạn node

Mỗi token gửi tới **≤ 4 node**. Trung bình **3.2 expert/node**:

$$4 \text{ node} \times 3.2 \text{ expert/node} = 12.8 \approx \mathbf{13 \text{ expert}}$$

> Nghĩa là: dù V3 chỉ chọn **8 expert**, kiến trúc này **có thể scale tới 13 expert mà chi phí giao tiếp KHÔNG đổi**. Còn dư địa lớn.

### Tối ưu SM

Dùng **warp specialization**: chia **20 SM** (trên 132 SM của H800) thành 10 kênh; số warp cho mỗi tác vụ (IB gửi / IB→NVLink / NVLink nhận) điều chỉnh động. Dùng **PTX tùy biến** + auto-tune chunk size → giảm dùng L2 cache, ít nhiễu SM khác.

> Kết quả: chỉ **20/132 SM** đủ bão hòa cả IB lẫn NVLink. (Đây cũng là lý do họ "than phiền" trong phần đề xuất phần cứng: dùng SM quý giá cho giao tiếp là lãng phí tensor core.)

---

## 10. GRPO — bóc tách hàm mục tiêu + ví dụ nhóm reward

### Vấn đề của PPO (cách cũ)

PPO cần một **critic/value model** để ước lượng baseline (giá trị kỳ vọng). Critic này thường **to bằng policy model** → với mô hình 671B, nuôi thêm một critic 671B là **tốn gấp đôi** bộ nhớ + tính toán.

### Ý tưởng GRPO: bỏ critic, dùng nhóm làm baseline

Với mỗi câu hỏi $q$, sample **một nhóm G output** $\{o_1,...,o_G\}$, chấm điểm $\{r_1,...,r_G\}$, rồi dùng **thống kê nhóm** làm baseline.

### Hàm mục tiêu (bóc tách)

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\Big(\min\big(\rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i\big) - \beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{ref})\Big)\Big]$$

- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$: **tỷ lệ xác suất** mới/cũ. >1 nếu policy mới thích output này hơn.
- $A_i$: **advantage** (output này tốt hơn/kém hơn trung bình nhóm bao nhiêu).
- $\min(\cdot, \text{clip}(\cdot))$: cơ chế **clip** của PPO — chặn cập nhật quá mạnh (nếu $\rho_i$ lệch xa 1).
- $-\beta\,\mathbb{D}_{KL}$: phạt nếu policy trôi quá xa **reference model** (giữ ổn định, chống "quên").

### Advantage = chuẩn hóa trong nhóm

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

### VÍ DỤ SỐ (nhóm G=4)

Reward 4 output: $r = \{0.9,\ 0.2,\ 0.5,\ 0.8\}$.
- $\text{mean} = 0.6$.
- Độ lệch: $\{0.3, -0.4, -0.1, 0.2\}$; bình phương $\{0.09,0.16,0.01,0.04\}$, tổng 0.30, $/4 = 0.075$, $\text{std} = 0.274$.
- Advantage:
  - $A_1 = (0.9-0.6)/0.274 = \mathbf{+1.10}$ → output 1 **được củng cố mạnh**.
  - $A_2 = (0.2-0.6)/0.274 = \mathbf{-1.46}$ → output 2 **bị ức chế mạnh**.
  - $A_3 = (0.5-0.6)/0.274 = -0.37$ → hơi ức chế.
  - $A_4 = (0.8-0.6)/0.274 = +0.73$ → củng cố.

> **Mấu chốt:** Baseline (mean = 0.6) lấy **ngay từ nhóm**, không cần critic model. "Tốt hơn trung bình anh em của mình thì được thưởng." ⇒ tiết kiệm tài nguyên khổng lồ mà vẫn ổn định.

### Reward Model trong V3

- **Rule-based** (toán đóng khung kết quả, code chạy test) → **chống gian lận tuyệt đối**, ưu tiên dùng.
- **Model-based** (đáp án tự do) → RM huấn luyện từ checkpoint SFT, kèm **chain-of-thought dẫn tới reward** để giảm reward hacking.

---

## 11. Chi phí huấn luyện — kiểm chứng con số

Kiểm tra lại các con số trong bài có nhất quán không:

- **180K GPU-hours / 1T token**, cụm **2048 GPU**:
  $$180000 / 2048 = 87.9 \text{ giờ} = \mathbf{3.66 \text{ ngày}} \text{ / 1T token} \checkmark$$
- **14.8T token**: $87.9 \times 14.8 = 1300 \text{ giờ} \approx 54 \text{ ngày} < 2 \text{ tháng}$ $\checkmark$
- Tổng pre-train: $180\text{K} \times 14.8 = 2664\text{K}$ GPU-hours $\checkmark$
- Chi phí ($2/GPU-giờ): $2664\text{K} \times \$2 = \$5.328\text{M}$ $\checkmark$
- Tổng (+ context 119K + post-train 5K): $2788\text{K}$ GPU-hours $\times \$2 = \$5.576\text{M}$ $\checkmark$

> ⚠️ **Nhắc lại để không hiểu sai:** Đây **chỉ** là lần train chính thức cuối — **không** gồm R&D, ablation, thử kiến trúc, dữ liệu. Đừng nói "DeepSeek làm ra model frontier chỉ với $5.5M" mà không kèm chú thích này.

---

## 12. Long context (YaRN)

Sau pre-train ở context **4K**, mở rộng 2 giai đoạn (mỗi giai đoạn 1000 step):
- **4K → 32K** (seq length 32K, batch 1920).
- **32K → 128K** (seq length 128K, batch 480).

Dùng **YaRN** — kỹ thuật **nội suy/ngoại suy tần số RoPE** để mô hình "quen" với vị trí xa hơn dải đã train. Chỉ áp lên **decoupled shared key $\mathbf{k}^R$** (phần mang RoPE). Cấu hình: scale $s=40$, $\alpha=1$, $\beta=32$, hệ số $\sqrt{t}=0.1\ln s + 1$.

**Kết quả:** test "Needle In A Haystack" (giấu một câu thông tin trong văn bản dài, bắt mô hình tìm) tốt trên **toàn dải tới 128K** ⇒ mô hình thực sự dùng được context dài, không chỉ "nhận" được input dài.

---

## Tóm tắt: mỗi đổi mới phá vỡ một đánh đổi cố hữu

| Kỹ thuật | Đánh đổi cũ buộc phải chịu | Cách phá vỡ |
|---|---|---|
| **MLA** | KV cache nhỏ ⟺ chất lượng (MQA/GQA) | Nén hạng thấp + tách RoPE → nhỏ **và** tốt |
| **Aux-Loss-Free** | Cân bằng tải ⟺ chất lượng | Bias ngoài gradient → không xung đột |
| **MTP** | Tín hiệu train nhiều ⟺ chi phí suy luận | Vứt module lúc deploy → free |
| **FP8** | Tốc độ ⟺ ổn định số học | Fine-grained quant + FP32 accumulation |
| **DualPipe** | Song song ⟺ overhead giao tiếp | Overlap + 2 chiều → giấu giao tiếp |
| **GRPO** | RL chất lượng ⟺ chi phí critic | Baseline từ nhóm → bỏ critic |

> **Thông điệp xuyên suốt:** DeepSeek-V3 không phát minh một "viên đạn bạc" duy nhất. Sức mạnh của nó là **đồng thiết kế** — mỗi kỹ thuật giải một đánh đổi, và chúng cộng hưởng để cùng đạt mục tiêu "frontier với chi phí thấp".
