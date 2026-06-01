from __future__ import annotations

from pathlib import Path
import html
import re

from bs4 import BeautifulSoup
import markdown


OUT = Path("index.html")


ARTICLE_MD = r"""
# GIẢI THÍCH DEEPSEEK-V3 TỪ ĐẦU ĐẾN CUỐI

> Mục tiêu của trang này không phải là chép lại công thức. Mục tiêu là: nếu bạn chưa biết MoE, attention, FP8, pipeline hay RLHF là gì, bạn vẫn có thể đi từ ý tưởng gốc đến lý do vì sao DeepSeek-V3 chọn từng kỹ thuật.
>
> Nguyên tắc viết lại: giữ đầy đủ các con số và công thức quan trọng của bản Markdown gốc, nhưng mỗi ý đều được đặt vào bối cảnh: **vấn đề là gì**, **cách cũ kẹt ở đâu**, **DeepSeek làm gì**, **vì sao cách đó hợp lý**, và **đổi lại phải trả giá gì**.

## 0. Trước hết: DeepSeek-V3 đang cố giải bài toán gì?

Một mô hình ngôn ngữ lớn không chỉ cần “thông minh”. Nó còn phải **train được**, **chạy được**, và **không quá đắt**. Khi mô hình lên hàng trăm tỉ tham số, ba thứ sau thường va vào nhau:

- **Dung lượng kiến thức**: mô hình càng nhiều tham số thì càng có nhiều “chỗ” để lưu pattern ngôn ngữ, toán, code, tri thức.
- **Chi phí tính toán**: mỗi token đi qua càng nhiều tham số thì càng tốn GPU.
- **Bộ nhớ và giao tiếp**: mô hình càng lớn thì càng khó nhét lên GPU; MoE còn phải gửi token qua mạng tới đúng expert.

DeepSeek-V3 chọn hướng **đồng thiết kế**: không có một mẹo duy nhất cứu tất cả. Mỗi kỹ thuật xử lý một nút thắt:

| Nút thắt | Nếu làm theo cách đơn giản sẽ gặp gì? | DeepSeek-V3 dùng gì? |
|---|---|---|
| Muốn nhiều tham số nhưng compute không nổ | Dense model 671B sẽ quá đắt mỗi token | MoE thưa: 671B tổng, 37B kích hoạt |
| Context dài làm KV cache phình to | MHA 128K context cần hàng trăm GB cache | MLA nén KV cache |
| MoE dễ lệch tải expert | Một vài expert quá tải, GPU khác chờ | Auxiliary-loss-free bias + loss rất nhẹ |
| Train cần nhiều tín hiệu học hơn | Next-token prediction chỉ dạy một bước | MTP dự đoán thêm token tương lai |
| Train quá đắt nếu dùng BF16 hết | BF16 tốn bộ nhớ và băng thông | FP8 mixed precision có kiểm soát |
| Pipeline nhiều GPU bị chờ | Bubble làm GPU rảnh vô ích | DualPipe chồng lấp tính toán/giao tiếp |
| RLHF cần critic quá lớn | PPO cần value model gần bằng policy | GRPO dùng baseline từ nhóm output |
| Muốn context 128K | RoPE gốc không quen vị trí xa | YaRN mở rộng context theo giai đoạn |

Các hằng số sẽ xuất hiện xuyên suốt:

| Ký hiệu | Nghĩa | Giá trị trong V3 |
|---|---|---|
| $L$ | số lớp Transformer | 61 |
| $d$ | hidden dimension | 7168 |
| $n_h$ | số attention head | 128 |
| $d_h$ | chiều mỗi head | 128 |
| $d_c$ | chiều nén KV trong MLA | 512 |
| $d_c'$ | chiều nén query trong MLA | 1536 |
| $d_h^R$ | chiều nhánh RoPE tách riêng | 64 |
| $N_s$ | số shared expert | 1 |
| $N_r$ | số routed expert | 256 |
| $K_r$ | routed expert được chọn mỗi token | 8 |
| — | intermediate dimension mỗi expert | 2048 |
| $V$ | vocab size | 128K |
| — | tổng tham số / tham số kích hoạt | 671B / 37B |

### Ba khái niệm nền phải nắm

**Tham số** là các trọng số mô hình học được. Nhiều tham số hơn thường nghĩa là mô hình có nhiều dung lượng hơn, nhưng không tự động nghĩa là mọi tham số đều được dùng cho mọi token.

**Kích hoạt** nghĩa là phần mô hình thật sự chạy khi xử lý một token. Trong dense Transformer, gần như toàn bộ lớp đều chạy cho mọi token. Trong MoE thưa, chỉ một số expert chạy.

**Cache** là bộ nhớ lưu lại thứ đã tính ở các token trước. Khi sinh token thứ 1000, mô hình không muốn tính lại key/value của 999 token cũ, nên lưu chúng vào KV cache. Context càng dài, cache càng lớn.

Nếu nhớ ba ý này, bạn sẽ hiểu vì sao DeepSeek-V3 liên tục tách “có nhiều” khỏi “phải tính nhiều”: nhiều tham số nhưng ít tham số kích hoạt; context dài nhưng KV cache nhỏ; train thêm mục tiêu phụ nhưng deploy có thể bỏ module phụ.

## 1. Vì sao có con số 671B / 37B?

Nghe “671B tham số nhưng chỉ kích hoạt 37B mỗi token” rất dễ hiểu sai thành “mô hình chỉ dùng một phần nhỏ nên phần còn lại vô nghĩa”. Không phải vậy. Cách đúng để nhìn là: **671B là kho chuyên gia**, còn **37B là nhóm chuyên gia được gọi cho một token cụ thể**.

### Cách cũ: dense model

Ở mô hình dense, mỗi token đi qua gần như toàn bộ tham số của mỗi lớp. Nếu làm dense 671B, mỗi token phải dùng compute tương ứng với 671B tham số. Điều này mạnh nhưng cực đắt:

- Train chậm hơn vì mỗi token phải nhân ma trận khổng lồ.
- Inference đắt hơn vì mỗi token sinh ra đều đi qua toàn bộ mô hình.
- Bộ nhớ GPU vẫn phải chứa toàn bộ tham số.

Cách này hợp lý nếu có ngân sách cực lớn và muốn hệ thống đơn giản. Nhưng nếu mục tiêu là model frontier với chi phí thấp hơn, dense 671B không phải lựa chọn tốt.

### DeepSeek chọn MoE thưa

MoE, viết tắt của Mixture-of-Experts, thay một FFN khổng lồ bằng nhiều expert nhỏ hơn. Mỗi token không đi qua tất cả expert, mà router chọn vài expert phù hợp nhất.

Trong DeepSeek-V3:

- 3 lớp đầu là dense.
- 58 lớp còn lại là MoE.
- Mỗi lớp MoE có 256 routed expert và 1 shared expert.
- Mỗi token đi qua 8 routed expert + 1 shared expert.

Vì vậy, mô hình có rất nhiều expert để lưu tri thức, nhưng một token chỉ gọi một nhóm nhỏ expert.

### Tính tham số của một expert

Mỗi expert là FFN kiểu SwiGLU, có ba ma trận chính: `gate`, `up`, `down`. Với $d=7168$ và $d_{ff}=2048$:

$$\text{params/expert} = 3 \times d \times d_{ff} = 3 \times 7168 \times 2048 \approx 44.0\text{ triệu}$$

Vì mỗi lớp MoE có 256 routed expert:

$$\text{params routed/lớp} = 256 \times 44.0\text{M} \approx 11.3\text{B}$$

Với 58 lớp MoE:

$$11.3\text{B} \times 58 \approx 654\text{B}$$

Cộng thêm attention, embedding, 3 lớp dense, shared experts và các phần còn lại thì ra khoảng **671B tham số**.

### Tại sao mỗi token chỉ kích hoạt khoảng 37B?

Mỗi token chỉ chạy 8 routed expert và 1 shared expert:

$$\text{params kích hoạt/lớp MoE} = 9 \times 44.0\text{M} \approx 396\text{M}$$

Qua 58 lớp MoE:

$$396\text{M} \times 58 \approx 23\text{B}$$

Cộng thêm attention, embedding và dense layers thì thành khoảng **37B tham số kích hoạt mỗi token**.

### Vì sao cách này tốt hơn dense cùng kích thước?

MoE tách hai thứ vốn bị trói vào nhau trong dense model:

- **Dung lượng lưu tri thức** gần với model 671B, vì tổng tham số thật là 671B.
- **Chi phí xử lý mỗi token** gần với model nhỏ hơn nhiều, vì chỉ 37B tham số được kích hoạt.

Nói ngắn: dense model bắt bạn “mở cả thư viện” cho mỗi câu hỏi; MoE cho bạn gọi đúng vài chuyên gia liên quan.

### Vì sao không chọn ít expert hơn hoặc kích hoạt nhiều expert hơn?

Nếu ít expert hơn, mô hình mất dung lượng chuyên môn hóa. Mỗi expert phải học quá nhiều thứ, giống một người phải làm mọi nghề.

Nếu kích hoạt quá nhiều expert, chất lượng có thể tăng nhưng compute và giao tiếp tăng theo. Khi token phải đi tới nhiều expert trên nhiều GPU, chi phí mạng sẽ ăn mất lợi ích.

Con số 256 routed expert và top-8 là điểm cân bằng: đủ nhiều expert để chuyên môn hóa, nhưng mỗi token chỉ dùng một phần nhỏ để giữ chi phí thấp.

### Đổi lại là gì?

MoE không miễn phí. Nó đổi compute lấy độ phức tạp hệ thống:

- Phải chứa toàn bộ 671B tham số trong bộ nhớ phân tán.
- Phải có router chọn expert đúng.
- Phải cân bằng tải, nếu không một số expert/GPU quá tải.
- Phải tối ưu all-to-all communication, vì token có thể phải đi qua mạng tới expert.

Vì vậy các phần sau của DeepSeek-V3 gần như đều là để “trả nợ kỹ thuật” cho quyết định dùng MoE lớn.

## 2. MLA: vì sao attention cần nén KV cache?

Attention giúp token hiện tại nhìn lại các token trước. Khi sinh văn bản, mô hình đã xử lý các token cũ rồi, nên nó lưu key/value của chúng vào **KV cache**. Vấn đề: context càng dài thì KV cache càng phình.

### Cách cũ: Multi-Head Attention đầy đủ

Trong MHA chuẩn, mỗi token ở mỗi lớp cần cache:

$$K + V = 2 \times n_h \times d_h = 2 \times 128 \times 128 = 32768\text{ phần tử}$$

Với context 128K và 61 lớp, chỉ riêng KV cache BF16 đã khoảng:

$$32768 \times 61 \times 128000 \times 2 \approx 512\text{ GB}$$

512GB chỉ cho KV cache là bất khả thi với một GPU 80GB, và còn rất nặng ngay cả khi phân tán.

### Vì sao không dùng MQA/GQA cho đơn giản?

MQA/GQA giảm cache bằng cách chia sẻ key/value giữa nhiều head. Cách này hiệu quả, nhưng nó giảm độ tự do của attention: nhiều head phải nhìn bằng cùng key/value hơn. Khi model lớn và yêu cầu chất lượng cao, giảm quá mạnh có thể làm mất thông tin.

DeepSeek muốn cache nhỏ nhưng vẫn giữ chất lượng gần MHA. Vì vậy họ chọn cách nén hạng thấp: không ép các head dùng chung KV thô, mà lưu một vector tiềm ẩn nhỏ rồi giải nén khi cần.

### MLA làm gì?

MLA lưu key/value dưới dạng vector nén:

$$\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t$$

Ở đây:

- $\mathbf{h}_t \in \mathbb{R}^{7168}$ là hidden state của token.
- $W^{DKV} \in \mathbb{R}^{512 \times 7168}$ nén hidden state xuống 512 chiều.
- $\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$ là thứ chính cần cache.

Khi cần key/value đầy đủ, mô hình giải nén:

$$\mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}, \qquad \mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}$$

Điểm quan trọng: cache lưu vector nén, không lưu toàn bộ $K,V$ đã giải nén.

### Vì sao phải tách RoPE?

RoPE đưa thông tin vị trí vào query/key bằng phép quay phụ thuộc vị trí. Nếu trộn RoPE vào vector nén chung, phép giải nén sẽ bị phụ thuộc vị trí và khó “absorb” vào tính toán hiệu quả.

DeepSeek tách một nhánh nhỏ cho vị trí:

$$\mathbf{k}_t^R = \operatorname{RoPE}(W^{KR}\mathbf{h}_t)$$

với $\mathbf{k}_t^R \in \mathbb{R}^{64}$. Key cuối cùng cho head $i$ là:

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C;\mathbf{k}_t^R]$$

Tức là:

- phần nội dung được nén mạnh qua $\mathbf{c}_t^{KV}$;
- phần vị trí được giữ riêng trong $\mathbf{k}_t^R$.

### Tiết kiệm bao nhiêu?

| Cách | Mỗi token mỗi lớp cache gì? | Số phần tử |
|---|---|---|
| MHA chuẩn | K đầy đủ + V đầy đủ | $2 \times 128 \times 128 = 32768$ |
| MLA | $\mathbf{c}_t^{KV}$ + $\mathbf{k}_t^R$ | $512 + 64 = 576$ |

Tỷ lệ giảm:

$$\frac{32768}{576} \approx 56.9\times$$

Với 128K context, 61 lớp, BF16:

- MLA: $576 \times 61 \times 128000 \times 2 \approx 9\text{ GB}$.
- MHA: khoảng $512\text{ GB}$.

### Vì sao MLA tốt hơn “chỉ giảm số head”?

Giảm số head hoặc chia sẻ KV làm mô hình ít cách nhìn hơn. MLA giữ cấu trúc nhiều head nhưng nén phần được cache. Nó giống như lưu file ở dạng nén tốt, khi cần thì giải nén đúng cách, thay vì vứt bớt nội dung.

Đổi lại, MLA phải thêm ma trận nén/giải nén và thiết kế nhánh RoPE riêng. Nhưng cái giá đó nhỏ hơn rất nhiều so với việc giữ KV cache khổng lồ.

## 3. DeepSeekMoE gating: router chọn expert như thế nào?

Khi dùng MoE, câu hỏi không còn là “mỗi token đi qua FFN nào?” mà là “token này nên gửi tới expert nào?”. Đó là việc của **router** hay **gating**.

### Output của lớp MoE

Công thức:

$$\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s}\operatorname{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t}\operatorname{FFN}_i^{(r)}(\mathbf{u}_t)$$

Đọc theo nghĩa thường:

- $\mathbf{u}_t$ là input đi qua residual.
- Shared expert luôn chạy, để giữ năng lực chung.
- Routed experts chỉ chạy nếu router chọn.
- $g_{i,t}$ là trọng số đóng góp của expert $i$ cho token $t$.

### Router tính điểm ra sao?

Mỗi expert có một vector đại diện $\mathbf{e}_i$. Token có vector $\mathbf{u}_t$. Độ hợp giữa token và expert:

$$s_{i,t} = \operatorname{Sigmoid}(\mathbf{u}_t^\top \mathbf{e}_i)$$

Nếu $s_{i,t}$ cao, token hợp với expert đó hơn.

Sau đó chọn top-8:

$$g'_{i,t} =
\begin{cases}
s_{i,t} & \text{nếu } s_{i,t}\text{ thuộc top-}K\\
0 & \text{ngược lại}
\end{cases}
$$

Rồi chuẩn hóa:

$$g_{i,t} = \frac{g'_{i,t}}{\sum_j g'_{j,t}}$$

Chuẩn hóa để tổng trọng số của các expert được chọn bằng 1, tránh output phình tùy số điểm.

### Ví dụ nhỏ

Giả sử có 6 expert, chọn top-2, điểm:

$$s = [0.8, 0.1, 0.6, 0.3, 0.7, 0.2]$$

Top-2 là expert 1 và 5. Trọng số:

$$g_1 = \frac{0.8}{0.8+0.7}=0.533,\qquad g_5=\frac{0.7}{1.5}=0.467$$

Output dùng shared expert và hai routed expert đó.

### Vì sao dùng sigmoid chứ không softmax toàn bộ?

Softmax khiến các expert cạnh tranh trực tiếp: một expert tăng điểm thì các expert khác bị ép giảm xác suất tương đối. Sigmoid cho điểm từng expert độc lập hơn; một token có thể “hợp” với nhiều expert cùng lúc, rồi top-K quyết định chọn.

Điều này hợp với MoE vì token thường có nhiều khía cạnh: một đoạn code toán có thể cần expert code và expert toán. Sigmoid cho router biểu diễn sự phù hợp đa chiều trước khi cắt top-K.

### Vì sao vẫn cần top-K?

Nếu không top-K, token đi qua quá nhiều expert, MoE mất lợi ích sparse. Nếu top-K quá nhỏ, router có thể thiếu chuyên gia cần thiết. Top-8 là cách giữ compute thấp nhưng vẫn cho token phối hợp nhiều expert.

## 4. Auxiliary-loss-free: vì sao phải cân bằng tải expert?

MoE chỉ hiệu quả nếu các expert được dùng tương đối cân bằng. Nếu 1 expert nhận 400 token còn expert khác nhận 5 token, GPU chứa expert đầu sẽ quá tải, GPU kia chờ. Tổng tốc độ bị quyết định bởi GPU chậm nhất.

### Cách cũ: thêm auxiliary loss

Cách phổ biến là thêm một loss phụ để phạt routing lệch:

$$\mathcal{L}_{aux} = \alpha \sum_i f_i P_i$$

Ý tưởng: ép router chọn đều hơn. Vấn đề: loss phụ đi vào gradient của router. Nó không chỉ nói “đừng quá lệch tải”, mà còn kéo cách router học từ dữ liệu ngôn ngữ.

Nếu $\alpha$ lớn, tải cân bằng nhưng chất lượng có thể giảm. Nếu $\alpha$ nhỏ, chất lượng ít bị ảnh hưởng nhưng không đủ chống collapse. Đây là đánh đổi khó.

### DeepSeek làm gì khác?

DeepSeek thêm bias $b_i$ vào **bước chọn expert**, nhưng không đưa bias vào trọng số output:

$$g'_{i,t} =
\begin{cases}
s_{i,t} & \text{nếu } s_{i,t}+b_i\text{ thuộc top-}K\\
0 & \text{ngược lại}
\end{cases}
$$

Bias chỉ giúp expert thiếu tải dễ được chọn hơn, expert quá tải khó được chọn hơn. Nhưng khi expert đã được chọn, output vẫn nhân với $s_{i,t}$ gốc, không nhân với $s_{i,t}+b_i$.

### Vì sao chi tiết này quan trọng?

Nếu bias đi vào output, nó sẽ bóp méo tín hiệu chuyên môn: expert được chọn không phải vì thật sự phù hợp mà vì được “thưởng điểm”. DeepSeek chỉ dùng bias để điều phối giao thông, không dùng nó để thay đổi chuyên môn của expert.

Luật cập nhật:

$$b_i \leftarrow b_i - \gamma \quad \text{nếu expert }i\text{ quá tải}$$

$$b_i \leftarrow b_i + \gamma \quad \text{nếu expert }i\text{ thiếu tải}$$

### Ví dụ

Giả sử 3 expert, tải lý tưởng là 33% mỗi expert. Step 1 đo được:

- A: 50%, quá tải.
- B: 30%, hơi thiếu.
- C: 20%, thiếu nhiều.

Với $\gamma=0.001$ và bias ban đầu $[0,0,0]$:

- $b_A=-0.001$
- $b_B=+0.001$
- $b_C=+0.001$

Một token có điểm gốc $s=[0.50,0.49,0.48]$. Điểm dùng để chọn:

$$s+b=[0.499,0.491,0.481]$$

A vẫn có thể thắng nếu thật sự phù hợp. Nhưng khoảng cách thu hẹp, nên các token ranh giới sẽ dần chuyển sang B/C. Đây là điều ta muốn: cân bằng tải bằng các trường hợp ranh giới, không phá lựa chọn rõ ràng.

### Tại sao tốt hơn auxiliary loss?

Vì tín hiệu cân bằng tải không trộn trực tiếp vào gradient học ngôn ngữ. Router vẫn học expert nào hợp token nào; bias chỉ là cơ chế điều tiết bên ngoài. Bản gốc ghi nhận ablation aux-loss-free tốt hơn auxiliary loss trên nhiều benchmark, ví dụ GSM8K từ 70.7 lên 74.5 trong so sánh được nêu.

### Vì sao cuối training có thể đặt $\gamma=0$?

Khi tải đã ổn định, tiếp tục cập nhật bias có thể gây dao động nhỏ không cần thiết. Đặt $\gamma=0$ trong 500B token cuối giúp khóa trạng thái cân bằng đã học.

## 5. Sequence-wise balance loss: nếu đã có bias, vì sao vẫn cần loss phụ?

Auxiliary-loss-free cân bằng tải ở mức hệ thống/batch. Nhưng vẫn có thể xảy ra trường hợp một **chuỗi riêng lẻ** dùng expert quá lệch. Điều này không nhất thiết làm hỏng toàn hệ thống, nhưng có thể làm một sequence học kém hoặc routing quá tập trung.

DeepSeek giữ một loss rất nhẹ:

$$\mathcal{L}_{Bal} = \alpha \sum_{i=1}^{N_r} f_i P_i$$

với $\alpha=0.0001$.

Tần suất expert được chọn:

$$f_i = \frac{N_r}{K_r T}\sum_{t=1}^{T}\mathbb{1}(\text{expert }i\in\text{Top-K của token }t)$$

Điểm hợp trung bình:

$$P_i = \frac{1}{T}\sum_{t=1}^{T}s'_{i,t},\qquad s'_{i,t}=\frac{s_{i,t}}{\sum_j s_{j,t}}$$

### Công thức này phạt điều gì?

Nó phạt expert vừa được chọn nhiều vừa có điểm cao trong cùng sequence. Nếu một expert thống trị quá mạnh, $f_i$ và $P_i$ cùng cao, tích $f_iP_i$ cao.

### Vì sao $\alpha$ rất nhỏ?

Vì đây chỉ là dây an toàn. Nếu loss này mạnh, nó quay lại vấn đề cũ: ép router vì cân bằng hơn là vì chất lượng. DeepSeek chủ yếu dựa vào bias ngoài gradient, còn loss này chỉ ngăn các trường hợp lệch cục bộ.

Nói cách khác: bias là tay lái chính cho cân bằng tải; sequence-wise loss là gờ giảm tốc.

## 6. MTP: vì sao dự đoán thêm token tương lai giúp train?

Mô hình ngôn ngữ thường học bằng next-token prediction: tại vị trí $i$, dự đoán token $i+1$. Đây là mục tiêu đơn giản và rất mạnh. Nhưng nó có hạn chế: mỗi vị trí chỉ nhận tín hiệu trực tiếp từ một token kế tiếp.

### Ý tưởng của MTP

MTP, Multi-Token Prediction, yêu cầu mô hình dự đoán thêm token xa hơn. DeepSeek-V3 dùng $D=1$, nghĩa là ngoài dự đoán token kế tiếp của mô hình chính, có thêm module dự đoán thêm một token tương lai nữa.

Vì sao có ích?

- Tín hiệu học dày hơn: một vị trí không chỉ học “từ kế tiếp là gì” mà còn học một phần hướng đi tiếp theo.
- Biểu diễn có xu hướng “lên kế hoạch” tốt hơn, vì phải chứa thông tin hữu ích cho nhiều bước.
- Khi deploy, module phụ có thể bỏ đi, nên chi phí inference chính không tăng.

### Module MTP hoạt động thế nào?

Với module thứ $k$:

$$\mathbf{h}_i'^k = M_k[\operatorname{RMSNorm}(\mathbf{h}_i^{k-1});\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]$$

Giải thích:

- $\mathbf{h}_i^{k-1}$ là biểu diễn đã có từ model chính hoặc module trước.
- $\operatorname{Emb}(t_{i+k})$ là embedding của token tương lai gần hơn, dùng để duy trì chuỗi nhân quả.
- Hai vector được ghép lại, rồi $M_k$ chiếu từ $2d$ về $d$.

Sau đó:

$$\mathbf{h}_i^k = \operatorname{TRM}_k(\mathbf{h}_i'^k)$$

và:

$$P_{i+k+1}^k = \operatorname{OutHead}(\mathbf{h}_i^k)$$

Embedding và output head được chia sẻ với model chính, giúp module phụ không học một không gian từ vựng riêng lệch khỏi model chính.

### Loss MTP

$$\mathcal{L}_{MTP}^{k} = -\frac{1}{T}\sum_i \log P_i^k[t_i]$$

$$\mathcal{L}_{MTP} = \frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{MTP}^{k}$$

DeepSeek đặt $\lambda=0.3$ cho 10T token đầu, rồi $\lambda=0.1$ cho 4.8T token sau.

### Vì sao không dùng nhiều head song song độc lập?

Nếu mỗi head dự đoán một token tương lai độc lập từ cùng biểu diễn, chúng có thể thiếu quan hệ nhân quả giữa các bước. DeepSeek dùng các module tuần tự: module sau dựa vào module trước. Điều này giữ “dòng thời gian” hợp lý hơn.

### Khi inference thì sao?

Có hai lựa chọn:

1. **Bỏ module MTP**. Model chính chạy như bình thường, chi phí inference không tăng. Đây là lý do MTP hấp dẫn: nó giúp train nhưng có thể không làm deploy nặng hơn.
2. **Dùng cho speculative decoding**. MTP đoán trước token, model chính xác minh. Nếu tỷ lệ chấp nhận cao, tốc độ sinh token tăng. Bản gốc nêu tỷ lệ chấp nhận 85–90% và TPS khoảng $\times 1.8$.

### Đổi lại là gì?

Train phức tạp hơn, thêm module và loss phụ. Nếu $\lambda$ quá lớn, mục tiêu phụ có thể kéo model khỏi next-token objective chính. Vì vậy DeepSeek giảm $\lambda$ về 0.1 ở giai đoạn sau.

## 7. FP8: vì sao dùng số 8-bit mà không làm training sập?

Huấn luyện LLM chủ yếu là nhân ma trận khổng lồ. Nếu dùng BF16, mỗi số tốn 16 bit. FP8 chỉ 8 bit, nghĩa là có thể giảm băng thông/bộ nhớ và tăng tốc GEMM. Nhưng FP8 rất dễ làm training bất ổn.

### FP8 E4M3 là gì?

E4M3 có:

- 1 bit dấu.
- 4 bit exponent.
- 3 bit mantissa.

So với BF16, mantissa ít hơn nhiều, nên độ phân giải thô hơn. Nói đơn giản: FP8 nhớ số “ít chữ số” hơn. Nếu scale không đúng, số nhỏ bị mất, số lớn bị cắt, gradient nhiễu và model có thể phân kỳ.

### Vì sao per-tensor quantization nguy hiểm?

Cách ngây thơ: lấy max tuyệt đối của cả tensor để scale. Giả sử tensor đa số nằm trong $[-2,2]$, nhưng có một outlier bằng 1000.

Scale:

$$448/1000 = 0.448$$

Một giá trị bình thường $0.5$ thành $0.224$ sau scale. Giá trị nhỏ $0.01$ thành $0.00448$, rất dễ bị làm tròn mất. Chỉ một outlier đã kéo thang đo của toàn tensor, làm hỏng phần lớn giá trị bình thường.

### DeepSeek dùng fine-grained quantization

Thay vì một scale cho cả tensor, chia thành tile nhỏ:

- Activation: tile $1\times128$.
- Weight: tile $128\times128$.

Nếu outlier nằm ở một tile, chỉ tile đó bị ảnh hưởng. Các tile khác giữ scale phù hợp. Đây là lý do “fine-grained” tốt hơn: nó cô lập outlier.

### Vì sao không dùng INT8?

INT8 cần scale và thường phù hợp hơn cho inference hoặc một số phần đã ổn định. Training cần gradient, activation, weight thay đổi liên tục và có dải động phức tạp. FP8 giữ exponent nên biểu diễn dải động tự nhiên hơn INT8. DeepSeek vẫn phải rất cẩn thận, nhưng FP8 là lựa chọn hợp lý hơn cho mixed precision training.

### Vấn đề tích lũy trên H800

Tensor Core khi cộng dồn FP8 GEMM trên Hopper chỉ giữ khoảng 14 bit trong một số bước tích lũy. Với chiều trong $K$ lớn, ví dụ $K=4096$, lỗi tích lũy có thể lên tới khoảng 2%, đủ làm hỏng training.

DeepSeek dùng “promotion to CUDA cores”: sau mỗi $N_C=128$ phần tử, đưa tổng trung gian sang thanh ghi FP32 trên CUDA Core để cộng chính xác hơn. $N_C=128$ tương ứng 4 WGMMA, đủ cải thiện mà không làm overhead quá lớn.

### Online quantization

Cách delayed scaling dùng lịch sử max-abs từ các vòng trước để đoán scale. Nhưng activation hiện tại có thể khác lịch sử. DeepSeek dùng online quantization: tính max-abs tại chỗ cho tile/block hiện tại, scale theo dữ liệu thật đang xử lý.

### Vì sao activation gradient đặc biệt nguy hiểm?

Bản gốc nêu thử nghiệm lượng tử hóa block-wise $128\times128$ cho activation gradient (Dgrad) làm mô hình khoảng 16B phân kỳ sau khoảng 300B token.

Lý do: activation gradient có outlier tương quan theo token. Nếu gom nhiều token vào một block lớn, một token outlier kéo scale của token khác. Gradient sai lan ngược về các lớp nông, nên lỗi tích lũy rất nhanh. Vì vậy activation dùng tile $1\times128$ là điều kiện sống còn, không phải tối ưu nhỏ.

### Kết quả cuối

Sai số loss FP8 so với BF16 dưới 0.25%, trong vùng nhiễu ngẫu nhiên, được kiểm chứng trên model khoảng 16B và 230B. Đổi lại, GEMM nhanh hơn khoảng 2 lần và bộ nhớ activation giảm mạnh.

Ý chính: FP8 không tốt vì “8 bit là đủ”. FP8 tốt vì DeepSeek xây cả bộ cơ chế để làm 8 bit không phá training.

## 8. DualPipe: vì sao pipeline cần hai chiều?

Khi model quá lớn, ta chia các lớp lên nhiều GPU theo pipeline parallelism. Stage 1 xử lý xong mới đưa cho stage 2, stage 2 đưa cho stage 3, v.v.

Vấn đề gọi là **bubble**: có lúc GPU không có việc vì đang chờ stage khác.

### Cách cũ: 1F1B

1F1B xen kẽ forward và backward để giảm chờ, nhưng vẫn có bubble:

$$\text{bubble}_{1F1B} = (PP-1)(F+B)$$

Với $PP$ là số pipeline stage, $F$ là thời gian forward, $B$ là backward.

### ZB1P giảm bubble bằng cách tách weight backward

$$\text{bubble}_{ZB1P} = (PP-1)(F+B-2W)$$

Nó tốt hơn, nhưng DeepSeek còn có vấn đề đặc biệt: MoE tạo all-to-all communication nặng. Nếu không giấu giao tiếp, GPU vẫn chờ mạng.

### DualPipe làm gì?

DualPipe chạy pipeline hai chiều và chia chunk thành phần tính toán/giao tiếp để overlap:

- attention;
- all-to-all dispatch;
- MLP expert;
- all-to-all combine.

Công thức bubble:

$$\text{bubble}_{DualPipe} = \left(\frac{PP}{2}-1\right)(F\&B+B-3W)$$

Ví dụ minh họa với $PP=16$, $F=1$, $B=2$, $W=1$, $F\&B\approx2$:

- 1F1B: $(16-1)(1+2)=45$.
- ZB1P: $(16-1)(1+2-2)=15$.
- DualPipe: $(16/2-1)(2+2-3)=7$.

Con số này không phải đo thực, nhưng cho thấy thứ tự độ lớn: DualPipe giảm bubble rõ rệt.

### Vì sao hai chiều giúp?

Pipeline một chiều giống dây chuyền chỉ đẩy việc từ trái sang phải. Đầu và cuối dây chuyền dễ có lúc rỗng. Hai chiều cho micro-batch đi từ cả hai đầu, nên các stage có cơ hội nhận việc đều hơn.

Quan trọng hơn, DeepSeek dùng lịch để giao tiếp của phần này bị che sau tính toán của phần khác. Với MoE, all-to-all nặng gần bằng tính toán, nên overlap là bắt buộc.

### Đổi lại là gì?

DualPipe giữ hai bản sao tham số do hai chiều pipeline. Điều này tăng bộ nhớ, nhưng DeepSeek dùng expert parallelism lớn nên tham số đã được chia nhỏ; overhead không còn là điểm nghẽn chính.

Vì vậy DualPipe là lựa chọn hợp lý trong hệ thống này, nhưng không nhất thiết là lựa chọn đơn giản nhất cho mọi model nhỏ.

## 9. All-to-all: vì sao MoE bị nghẽn mạng?

Trong MoE, expert nằm trên nhiều GPU. Token ở GPU này có thể được router gửi tới expert trên GPU khác. Khi mọi GPU đều gửi token cho mọi GPU, ta có all-to-all communication.

Nếu không tối ưu, MoE thắng về compute nhưng thua vì mạng.

### Hai loại kết nối

| Mạng | Phạm vi | Băng thông |
|---|---|---|
| NVLink | trong một node 8 GPU | 160 GB/s |
| InfiniBand | giữa các node | 50 GB/s |

NVLink nhanh hơn:

$$160/50 = 3.2\times$$

### Vì sao “IB trước, NVLink sau”?

DeepSeek gửi token qua InfiniBand tới GPU cùng in-node index ở node đích, rồi dùng NVLink trong node để chuyển tới GPU chứa expert. Vì NVLink nhanh hơn, phần chuyển nội bộ có thể overlap với IB.

Điều này hợp lý vì ta muốn dùng đường chậm nhất (IB) một cách có kiểm soát và dùng đường nhanh hơn (NVLink) để phân phối nội bộ.

### Giới hạn node và số expert

Bản gốc nêu mỗi token gửi tới không quá 4 node, trung bình 3.2 expert/node:

$$4 \times 3.2 = 12.8 \approx 13\text{ expert}$$

V3 chỉ chọn 8 routed expert, nên cấu trúc này còn dư địa về mặt giao tiếp.

### Vì sao cần kernel tùy chỉnh?

All-to-all mặc định có thể dùng nhiều tài nguyên và đụng vào compute. DeepSeek dùng warp specialization, dành khoảng 20/132 SM của H800 cho giao tiếp, chia thành 10 kênh và auto-tune chunk size.

Mục tiêu không phải chỉ là “gửi nhanh”, mà là “gửi nhanh trong khi phần tính toán chính vẫn chạy”. Nếu giao tiếp chiếm quá nhiều SM hoặc L2 cache, nó phá lợi ích của overlap.

### Bài học phần cứng

DeepSeek cũng đề xuất phần cứng tương lai nên hỗ trợ giao tiếp/quantization tốt hơn, vì hiện tại phải dùng SM quý giá để làm việc giao tiếp. Đây là ví dụ rõ của co-design: thuật toán tạo nhu cầu mới cho phần cứng.

## 10. GRPO: vì sao bỏ critic trong RL?

Sau pre-training và supervised fine-tuning, model thường được cải thiện bằng RL từ reward. Cách phổ biến là PPO. PPO cần một critic/value model để ước lượng baseline: output này tốt hơn kỳ vọng bao nhiêu?

### Vấn đề của critic

Với model nhỏ, thêm critic còn chịu được. Với policy 671B, critic cùng cỡ là gánh nặng khổng lồ: thêm bộ nhớ, thêm compute, thêm độ phức tạp train.

Vì vậy DeepSeek dùng GRPO, Group Relative Policy Optimization.

### Ý tưởng GRPO

Với mỗi câu hỏi $q$, sample một nhóm output:

$$\{o_1,o_2,\ldots,o_G\}$$

Chấm reward:

$$\{r_1,r_2,\ldots,r_G\}$$

Thay vì hỏi critic “kỳ vọng là bao nhiêu?”, dùng chính trung bình nhóm làm baseline. Advantage:

$$A_i = \frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)}$$

Output tốt hơn trung bình nhóm có advantage dương; kém hơn có advantage âm.

### Hàm mục tiêu

$$\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)-\beta D_{KL}(\pi_\theta\|\pi_{ref})\right)\right]$$

Trong đó:

- $\rho_i=\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ là tỷ lệ xác suất policy mới/cũ.
- Clip chặn policy thay đổi quá mạnh trong một bước.
- KL giữ policy không trôi quá xa model tham chiếu.

### Ví dụ

Reward nhóm 4 output:

$$r=\{0.9,0.2,0.5,0.8\}$$

Mean:

$$0.6$$

Std:

$$0.274$$

Advantage:

- $A_1=(0.9-0.6)/0.274=+1.10$
- $A_2=(0.2-0.6)/0.274=-1.46$
- $A_3=(0.5-0.6)/0.274=-0.37$
- $A_4=(0.8-0.6)/0.274=+0.73$

Như vậy model tăng xác suất output 1 và 4, giảm output 2 và 3.

### Vì sao không cần critic?

Vì baseline được lấy từ nhóm output cùng câu hỏi. Ta không cần một model riêng dự đoán giá trị tuyệt đối; chỉ cần biết trong nhóm này output nào tốt hơn tương đối.

Đổi lại, GRPO cần sample nhiều output cho mỗi câu hỏi. Nhưng chi phí đó thường hợp lý hơn việc nuôi thêm một critic cực lớn.

### Reward model của V3

Bản gốc nêu hai loại reward:

- Rule-based cho toán/code có đáp án rõ hoặc test chạy được. Cách này ít bị reward hacking hơn.
- Model-based cho đáp án tự do, dùng reward model từ checkpoint SFT và có chain-of-thought dẫn tới reward để giảm gian lận.

## 11. Chi phí huấn luyện: các con số có khớp không?

DeepSeek-V3 nêu chi phí training chính thức thấp so với quy mô model. Ta kiểm tra các phép tính:

Với 180K GPU-hours cho 1T token, cụm 2048 GPU:

$$180000/2048=87.9\text{ giờ}\approx3.66\text{ ngày}$$

Với 14.8T token:

$$87.9\times14.8=1300\text{ giờ}\approx54\text{ ngày}$$

Tổng pre-training:

$$180K\times14.8=2664K\text{ GPU-hours}$$

Nếu tính $2/GPU\text{-hour}$:

$$2664K\times2=5.328\text{ triệu USD}$$

Cộng context extension 119K GPU-hours và post-training 5K GPU-hours:

$$2788K\times2=5.576\text{ triệu USD}$$

### Vì sao con số này không nên bị hiểu sai?

Đây là chi phí của lần training chính thức được báo cáo, không bao gồm toàn bộ R&D, ablation, thử sai kiến trúc, chuẩn bị dữ liệu, nhân sự, hạ tầng, hoặc chi phí cơ hội. Vì vậy câu đúng là:

> DeepSeek-V3 báo cáo chi phí compute cho lần training chính thức khoảng 5.576M USD theo giả định $2/GPU-hour$.

Không nên nói đơn giản “làm model frontier chỉ tốn 5.5M” nếu không kèm điều kiện.

### Vì sao vẫn đáng chú ý?

Vì ngay cả khi không tính toàn bộ R&D, việc train một MoE 671B ổn định bằng FP8, DualPipe và hệ thống all-to-all tùy chỉnh vẫn chứng minh rằng kỹ thuật hệ thống có thể giảm chi phí rất lớn.

## 12. Long context YaRN: vì sao mở từ 4K lên 128K không chỉ là tăng số?

Transformer dùng RoPE để mã hóa vị trí. Nếu model chủ yếu train ở context 4K, nó không tự nhiên hiểu tốt vị trí 100K. Chỉ tăng giới hạn input không đảm bảo model dùng được context dài.

### DeepSeek mở context theo hai giai đoạn

Sau pre-training 4K:

1. 4K → 32K, sequence length 32K, batch 1920, 1000 step.
2. 32K → 128K, sequence length 128K, batch 480, 1000 step.

Dùng YaRN để nội suy/ngoại suy tần số RoPE, giúp model thích nghi với vị trí xa hơn.

Cấu hình:

$$s=40,\qquad \alpha=1,\qquad \beta=32,\qquad \sqrt{t}=0.1\ln s+1$$

### Vì sao chỉ áp lên $\mathbf{k}^R$?

Trong MLA, phần RoPE đã được tách vào decoupled shared key $\mathbf{k}^R$. Vì vậy muốn sửa hành vi vị trí, ta sửa đúng nhánh mang vị trí. Điều này sạch hơn việc đụng vào toàn bộ key/value nội dung.

### Kiểm tra bằng Needle In A Haystack

Needle test giấu một thông tin trong văn bản rất dài và hỏi model tìm lại. Nếu model chỉ “nhận” được input dài nhưng không thật sự truy hồi được, nó sẽ fail. Bản gốc nêu V3 đạt kết quả tốt trên toàn dải tới 128K, nghĩa là context dài không chỉ là thông số quảng cáo.

## 13. Kết luận: vì sao các kỹ thuật này phải đi cùng nhau?

Nếu chỉ nhìn từng kỹ thuật riêng lẻ, có thể nghĩ DeepSeek-V3 là tập hợp nhiều mẹo. Cách nhìn đúng hơn: mỗi kỹ thuật mở khóa kỹ thuật khác.

| Kỹ thuật | Đánh đổi cũ | Cách DeepSeek phá đánh đổi | Vì sao cần trong hệ thống chung |
|---|---|---|---|
| MoE | Nhiều tham số thì compute lớn | Sparse activation | Cho model có dung lượng 671B nhưng compute/token khoảng 37B |
| MLA | Context dài làm cache nổ | Cache vector nén + RoPE tách riêng | Làm 128K context thực tế hơn |
| Aux-loss-free | Cân bằng tải làm hại quality | Bias ngoài gradient | Giữ MoE chạy đều mà ít phá router |
| Sequence-wise loss | Một sequence có thể lệch expert | Loss rất nhẹ | Dây an toàn cục bộ |
| MTP | Nhiều tín hiệu train làm inference nặng | Module phụ có thể bỏ | Tăng chất lượng train mà deploy không phải trả đủ |
| FP8 | Train nhanh dễ mất ổn định | Fine-grained quant + FP32 accumulation | Giảm chi phí training quy mô lớn |
| DualPipe | Pipeline nhiều GPU có bubble | Pipeline hai chiều + overlap | Giữ GPU bận khi model phân tán |
| All-to-all kernel | MoE bị nghẽn mạng | Kernel giao tiếp tùy chỉnh | Biến MoE lớn thành thứ chạy được |
| GRPO | PPO cần critic lớn | Baseline từ nhóm | RL hậu huấn luyện rẻ hơn |
| YaRN | Context dài không tự nhiên tổng quát | Mở context theo giai đoạn | Cho model dùng được 128K context |

Thông điệp cuối cùng: DeepSeek-V3 không mạnh chỉ vì có 671B tham số. Nó mạnh vì kiến trúc, mục tiêu huấn luyện, số học FP8, pipeline, giao tiếp và hậu huấn luyện được thiết kế cùng nhau. Nếu bỏ một mắt xích, các mắt xích khác dễ mất tác dụng.
"""


def render_markdown_preserving_math(markdown_text: str) -> str:
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"@@MATH_{len(placeholders) - 1}@@"

    protected = re.sub(r"\$\$[\s\S]*?\$\$", stash, markdown_text)
    protected = re.sub(r"(?<!\$)\$(?!\$)(?:\\.|[^$])+\$(?!\$)", stash, protected)

    rendered = markdown.markdown(
        protected,
        extensions=["extra", "toc", "sane_lists", "smarty", "md_in_html"],
        extension_configs={"toc": {"permalink": False, "separator": "-"}},
        output_format="html5",
    )

    for index, math_text in enumerate(placeholders):
        rendered = rendered.replace(f"@@MATH_{index}@@", math_text)
    return rendered


def build_page() -> str:
    body = render_markdown_preserving_math(ARTICLE_MD)
    soup = BeautifulSoup(body, "html.parser")

    title = soup.find("h1").get_text(" ", strip=True)
    h2s = soup.find_all("h2")
    for index, h2 in enumerate(h2s, start=1):
        h2["data-section-index"] = f"{index:02d}"
        h2["class"] = (h2.get("class") or []) + ["section-heading"]

    for table in soup.find_all("table"):
        table["class"] = (table.get("class") or []) + ["data-table"]

    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text.startswith("$$") and text.endswith("$$"):
            p["class"] = (p.get("class") or []) + ["math-block"]

    def _clean_label(text: str) -> str:
        # Bỏ tiền tố "0. ", "1. " trùng với số thứ tự hiển thị riêng ở sidebar.
        return re.sub(r"^\s*\d+\.\s*", "", text)

    outline = "\n".join(
        f'<a href="#{html.escape(h2.get("id", ""))}">'
        f'<span>{html.escape(h2.get("data-section-index", ""))}</span>'
        f"<em>{html.escape(_clean_label(h2.get_text(' ', strip=True)))}</em></a>"
        for h2 in h2s
    )

    article_html = str(soup)
    section_count = len(h2s)
    h3_count = len(soup.find_all("h3"))
    table_count = len(soup.find_all("table"))
    math_count = ARTICLE_MD.count("$$")

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <meta name="description" content="Giải thích DeepSeek-V3 từ đầu đến cuối: MoE, MLA, FP8, DualPipe, GRPO, YaRN — bối cảnh, cách hoạt động và cái giá phải trả của từng kỹ thuật.">
  <meta name="color-scheme" content="light dark">
  <meta property="og:title" content="Giải thích DeepSeek-V3 từ đầu đến cuối">
  <meta property="og:description" content="Bản viết lại theo kiểu giáo trình về kiến trúc và huấn luyện DeepSeek-V3.">
  <meta property="og:type" content="article">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  {MATHJAX}
  <style>{CSS}</style>
</head>
<body>
  <div class="progress" aria-hidden="true"><span id="progress-bar"></span></div>

  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="#top">
        <span class="mark" aria-hidden="true">V3</span>
        <span class="brand-text">
          <strong>DeepSeek-V3</strong>
          <span>Giải thích kỹ thuật</span>
        </span>
      </a>
      <div class="toolbar">
        <a class="icon-btn" href="https://github.com/tson295/deepseek-v3-explained" target="_blank" rel="noopener" title="Mã nguồn trên GitHub" aria-label="Mã nguồn trên GitHub">
          <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
        </a>
        <button class="icon-btn" id="theme-toggle" type="button" title="Đổi giao diện sáng/tối" aria-label="Đổi giao diện sáng tối">
          <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
        </button>
        <button class="icon-btn" id="top-button" type="button" title="Lên đầu trang" aria-label="Lên đầu trang">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
        </button>
      </div>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar" aria-label="Mục lục">
      <div class="sidebar-inner">
        <p class="sidebar-title">Nội dung <span class="count">{section_count}</span></p>
        <nav class="outline-links">{outline}</nav>
      </div>
    </aside>

    <main id="top">
      <header class="hero">
        <span class="eyebrow">Phân tích kiến trúc &amp; huấn luyện</span>
        <h1>DeepSeek-V3, giải thích từ đầu đến cuối</h1>
        <p class="lead">Một bản viết lại theo kiểu giáo trình: mỗi kỹ thuật đều có bối cảnh, lý do xuất hiện, cách hoạt động, vì sao tốt hơn cách cũ và cái giá phải trả.</p>
        <dl class="stats" aria-label="Thông số chính">
          <div><dt>671B / 37B</dt><dd>Tổng tham số / kích hoạt mỗi token</dd></div>
          <div><dt>56.9&times;</dt><dd>Mức giảm KV cache của MLA so với MHA</dd></div>
          <div><dt>FP8</dt><dd>Mixed precision khi huấn luyện quy mô lớn</dd></div>
          <div><dt>128K</dt><dd>Context sau khi mở rộng bằng YaRN</dd></div>
        </dl>
      </header>

      <div class="note">
        <svg class="note-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
        <div><strong>Đây là bản viết lại phần giải thích, không chỉ convert Markdown.</strong> Trang gồm {section_count} phần chính, {h3_count} mục con, {table_count} bảng và {math_count // 2} khối công thức. Công thức LaTeX được bảo vệ trước khi render để Markdown không làm hỏng.</div>
      </div>

      <div class="mobile-outline">
        <details>
          <summary>Mục lục nhanh</summary>
          <nav class="outline-links">{outline}</nav>
        </details>
      </div>

      <article class="article" id="article">
{article_html}
      </article>

      <footer class="page-footer">
        <span>Biên soạn lại từ <strong>DeepSeek-V3 Technical Report</strong> · <a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noopener">arXiv:2412.19437</a> · công thức render bằng MathJax.</span>
        <a class="to-top" href="#top">↑ Lên đầu trang</a>
      </footer>
    </main>
  </div>

  <script>{SCRIPT}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Giao diện: tách CSS / JS / cấu hình MathJax ra biến chuỗi thô để dấu ngoặc
# nhọn không phải escape khi nhúng vào f-string ở build_page().
# ---------------------------------------------------------------------------

MATHJAX = r"""<script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] },
      svg: { fontCache: 'global' },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>"""


CSS = r"""
    :root {
      color-scheme: light;
      --bg: #fbfbfc;
      --surface: #ffffff;
      --surface-2: #f4f4f6;
      --surface-3: #ececef;
      --ink: #18181b;
      --text: #3f3f47;
      --muted: #71717a;
      --faint: #a1a1aa;
      --line: #e6e6ea;
      --line-2: #d6d6dc;
      --accent: #4f46e5;
      --accent-press: #4338ca;
      --accent-soft: #eef0ff;
      --accent-ring: rgba(79, 70, 229, .16);
      --code-bg: #1b1b22;
      --code-ink: #e7e7ef;
      --shadow-sm: 0 1px 2px rgba(24, 24, 27, .06), 0 1px 1px rgba(24, 24, 27, .04);
      --shadow: 0 4px 24px -8px rgba(24, 24, 27, .16);
      --radius: 12px;
      --radius-sm: 8px;
      --maxw: 1200px;
      --content: 768px;
    }

    [data-theme="dark"] {
      color-scheme: dark;
      --bg: #09090b;
      --surface: #111114;
      --surface-2: #19191d;
      --surface-3: #212127;
      --ink: #fafafa;
      --text: #c9c9d1;
      --muted: #8e8e98;
      --faint: #6a6a74;
      --line: #26262b;
      --line-2: #36363d;
      --accent: #8b8cf9;
      --accent-press: #a6a7fb;
      --accent-soft: rgba(129, 140, 248, .12);
      --accent-ring: rgba(139, 140, 249, .24);
      --code-bg: #0c0c10;
      --code-ink: #e3e3ec;
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, .4);
      --shadow: 0 8px 40px -12px rgba(0, 0, 0, .6);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; -webkit-text-size-adjust: 100%; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      font-size: 16.5px;
      line-height: 1.72;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }
    ::selection { background: var(--accent-ring); }
    a { color: var(--accent); }

    .progress { position: fixed; inset: 0 0 auto; height: 2px; z-index: 60; pointer-events: none; }
    .progress span { display: block; height: 100%; width: 0; background: var(--accent); }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 40;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--surface) 78%, transparent);
      backdrop-filter: blur(12px) saturate(1.4);
      -webkit-backdrop-filter: blur(12px) saturate(1.4);
    }
    .topbar-inner {
      width: min(var(--maxw), calc(100% - 48px));
      margin: 0 auto;
      height: 60px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .brand { display: flex; align-items: center; gap: 11px; text-decoration: none; color: inherit; min-width: 0; }
    .mark {
      width: 34px;
      height: 34px;
      flex: none;
      display: grid;
      place-items: center;
      border-radius: 9px;
      background: linear-gradient(150deg, var(--accent), var(--accent-press));
      color: #fff;
      font-weight: 800;
      font-size: .82rem;
      letter-spacing: -.02em;
      box-shadow: var(--shadow-sm);
    }
    .brand-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.18; }
    .brand-text strong { font-size: .96rem; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
    .brand-text span { font-size: .78rem; color: var(--muted); }
    .toolbar { display: flex; gap: 6px; align-items: center; }
    .icon-btn {
      width: 36px;
      height: 36px;
      display: inline-grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: var(--surface);
      color: var(--muted);
      cursor: pointer;
      text-decoration: none;
      transition: color .15s ease, border-color .15s ease, background .15s ease;
    }
    .icon-btn:hover { color: var(--ink); border-color: var(--line-2); background: var(--surface-2); }
    .icon-btn svg { width: 17px; height: 17px; display: block; }
    .icon-sun { display: none; }
    [data-theme="dark"] .icon-moon { display: none; }
    [data-theme="dark"] .icon-sun { display: block; }

    .layout {
      width: min(var(--maxw), calc(100% - 48px));
      margin: 0 auto;
      display: grid;
      grid-template-columns: 248px minmax(0, 1fr);
      gap: 56px;
      padding: 0 0 96px;
    }
    main { min-width: 0; grid-column: 2; grid-row: 1; }
    .sidebar { grid-column: 1; grid-row: 1; }

    .sidebar-inner {
      position: sticky;
      top: 84px;
      max-height: calc(100vh - 108px);
      overflow: auto;
      padding-right: 6px;
    }
    .sidebar-inner::-webkit-scrollbar { width: 8px; }
    .sidebar-inner::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px; }
    .sidebar-inner::-webkit-scrollbar-track { background: transparent; }
    .sidebar-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 0 0 14px;
      padding: 0 10px;
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .09em;
      text-transform: uppercase;
      color: var(--faint);
    }
    .sidebar-title .count {
      display: grid;
      place-items: center;
      min-width: 22px;
      height: 20px;
      padding: 0 6px;
      border-radius: 999px;
      background: var(--surface-2);
      color: var(--muted);
      font-size: .7rem;
    }
    .outline-links { display: flex; flex-direction: column; gap: 1px; border-left: 1px solid var(--line); }
    .outline-links a {
      position: relative;
      display: grid;
      grid-template-columns: 24px 1fr;
      gap: 9px;
      align-items: start;
      margin-left: -1px;
      padding: 7px 10px 7px 14px;
      border-left: 2px solid transparent;
      color: var(--muted);
      text-decoration: none;
      font-size: .855rem;
      line-height: 1.4;
      border-radius: 0 7px 7px 0;
      transition: color .15s ease, background .15s ease, border-color .15s ease;
    }
    .outline-links a > span { color: var(--faint); font-size: .76rem; font-weight: 600; font-variant-numeric: tabular-nums; padding-top: .08em; }
    .outline-links a > em { font-style: normal; }
    .outline-links a:hover { color: var(--ink); background: var(--surface-2); }
    .outline-links a.active { color: var(--accent); border-left-color: var(--accent); font-weight: 600; background: var(--accent-soft); }
    .outline-links a.active > span { color: var(--accent); }

    .hero { padding: 60px 0 38px; border-bottom: 1px solid var(--line); margin-bottom: 42px; }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-size: .76rem;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .eyebrow::before { content: ""; width: 20px; height: 1.5px; background: var(--accent); border-radius: 2px; }
    .hero h1 {
      margin: 18px 0 0;
      font-size: clamp(2.05rem, 4.6vw, 3.35rem);
      line-height: 1.07;
      letter-spacing: -.028em;
      font-weight: 800;
      color: var(--ink);
      max-width: 17ch;
    }
    .lead { margin: 18px 0 0; max-width: 60ch; font-size: clamp(1.02rem, 1.4vw, 1.18rem); line-height: 1.6; color: var(--muted); }
    .stats {
      display: flex;
      flex-wrap: wrap;
      margin: 36px 0 0;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--surface);
      box-shadow: var(--shadow-sm);
    }
    .stats > div { flex: 1 1 0; min-width: 150px; padding: 16px 20px; border-right: 1px solid var(--line); }
    .stats > div:last-child { border-right: 0; }
    .stats dt { font-size: clamp(1.2rem, 2vw, 1.5rem); font-weight: 800; letter-spacing: -.02em; color: var(--ink); font-variant-numeric: tabular-nums; }
    .stats dd { margin: 6px 0 0; font-size: .82rem; line-height: 1.45; color: var(--muted); }

    .note {
      display: flex;
      gap: 13px;
      align-items: flex-start;
      padding: 15px 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      color: var(--text);
      font-size: .92rem;
      line-height: 1.6;
      margin-bottom: 44px;
      box-shadow: var(--shadow-sm);
    }
    .note-ico { flex: none; width: 20px; height: 20px; margin-top: 2px; color: var(--accent); }
    .note strong { color: var(--ink); font-weight: 650; }

    .article { max-width: var(--content); }
    .article > h1:first-child { display: none; }
    .article h2, .article h3, .article h4 { color: var(--ink); letter-spacing: -.018em; scroll-margin-top: 84px; }
    .article h2 {
      margin: 3.6rem 0 1.3rem;
      padding-top: 2.3rem;
      border-top: 1px solid var(--line);
      font-size: clamp(1.5rem, 2.6vw, 1.95rem);
      line-height: 1.2;
      font-weight: 750;
    }
    .article h2:first-of-type { margin-top: .8rem; padding-top: 0; border-top: 0; }
    .article h3 { margin: 2.4rem 0 .8rem; font-size: clamp(1.12rem, 1.7vw, 1.34rem); line-height: 1.3; font-weight: 650; }
    .article h4 { margin: 1.9rem 0 .6rem; font-size: 1.05rem; font-weight: 650; }
    .article p, .article ul, .article ol, .article blockquote, .article pre, .article .table-wrap, .article .math-block { margin: 0 0 1.2rem; }
    .article p { color: var(--text); }
    .article a { color: var(--accent); text-decoration: none; border-bottom: 1px solid var(--accent-ring); transition: border-color .15s ease; }
    .article a:hover { border-bottom-color: var(--accent); }
    .article strong { color: var(--ink); font-weight: 650; }
    .article ul, .article ol { padding-left: 1.3rem; color: var(--text); }
    .article li { margin: .4rem 0; }
    .article li::marker { color: var(--faint); }

    .article blockquote { margin-left: 0; margin-right: 0; padding: 2px 0 2px 20px; border-left: 3px solid var(--accent); color: var(--muted); }
    .article blockquote p { color: var(--muted); }
    .article blockquote > :last-child { margin-bottom: 0; }

    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow-sm); }
    .data-table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: .9rem; }
    .data-table th, .data-table td { padding: 11px 16px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
    .data-table thead th { background: var(--surface-2); color: var(--ink); font-weight: 650; font-size: .76rem; letter-spacing: .04em; text-transform: uppercase; border-bottom: 1px solid var(--line-2); }
    .data-table td { color: var(--text); }
    .data-table tbody tr:last-child td { border-bottom: 0; }
    .data-table tbody tr:hover td { background: var(--surface-2); }

    .article code {
      font-family: "JetBrains Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
      font-size: .85em;
      padding: .14em .42em;
      border-radius: 5px;
      background: var(--surface-2);
      border: 1px solid var(--line);
      color: var(--ink);
    }
    .article pre {
      overflow-x: auto;
      padding: 18px 20px;
      border: 1px solid var(--line-2);
      border-radius: var(--radius);
      background: var(--code-bg);
      color: var(--code-ink);
      font-size: .86rem;
      line-height: 1.6;
    }
    .article pre code { padding: 0; border: 0; background: transparent; color: inherit; font-size: 1em; }
    .article hr { height: 1px; border: 0; background: var(--line); margin: 2.6rem 0; }

    .math-block {
      overflow-x: auto;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      text-align: center;
      color: var(--ink);
      box-shadow: var(--shadow-sm);
    }
    mjx-container { max-width: 100%; overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
    mjx-container[display="true"] { margin: 0; }

    .page-footer {
      max-width: var(--content);
      margin-top: 60px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: .85rem;
    }
    .page-footer strong { color: var(--text); font-weight: 600; }
    .page-footer a { color: var(--accent); text-decoration: none; }
    .page-footer a:hover { text-decoration: underline; }
    .page-footer .to-top { white-space: nowrap; font-weight: 600; }

    .mobile-outline { display: none; margin-bottom: 36px; }
    .mobile-outline details { border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); overflow: hidden; box-shadow: var(--shadow-sm); }
    .mobile-outline summary { cursor: pointer; padding: 14px 18px; font-weight: 650; color: var(--ink); list-style: none; display: flex; justify-content: space-between; align-items: center; }
    .mobile-outline summary::-webkit-details-marker { display: none; }
    .mobile-outline summary::after { content: "\\203A"; transform: rotate(90deg); color: var(--muted); transition: transform .2s ease; }
    .mobile-outline details[open] summary { border-bottom: 1px solid var(--line); }
    .mobile-outline details[open] summary::after { transform: rotate(-90deg); }
    .mobile-outline .outline-links { padding: 8px; border-left: 0; }
    .mobile-outline .outline-links a { border-left: 0; border-radius: 7px; }

    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; gap: 0; }
      .sidebar { display: none; }
      main { grid-column: 1; }
      .mobile-outline { display: block; }
      .article { max-width: none; }
      .page-footer { max-width: none; }
    }
    @media (max-width: 680px) {
      .topbar-inner, .layout { width: min(calc(100% - 32px), 100%); }
      body { font-size: 16px; }
      .brand-text span { display: none; }
      .hero { padding: 38px 0 30px; }
      .stats > div { flex-basis: 100%; border-right: 0; border-bottom: 1px solid var(--line); }
      .stats > div:last-child { border-bottom: 0; }
      .article h2 { margin-top: 2.8rem; }
    }
    @media print {
      body { background: #fff; color: #000; font-size: 11pt; }
      .progress, .topbar, .sidebar, .mobile-outline, .page-footer { display: none; }
      .layout { display: block; width: 100%; padding: 0; }
      main { grid-column: 1; }
      .hero { border: 0; padding: 0; margin-bottom: 1.2rem; }
      .stats { box-shadow: none; }
      .article { max-width: none; }
      .article > h1:first-child { display: block; }
      .article h2 { break-after: avoid; }
      .article a { color: #000; border: 0; }
    }
"""


SCRIPT = r"""
    const root = document.documentElement;
    const saved = localStorage.getItem('deepseek-v3-theme');
    if (saved) {
      root.dataset.theme = saved;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      root.dataset.theme = 'dark';
    }

    document.getElementById('theme-toggle').addEventListener('click', () => {
      const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('deepseek-v3-theme', next);
    });

    document.getElementById('top-button').addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    document.querySelectorAll('.article table').forEach((table) => {
      if (table.parentElement.classList.contains('table-wrap')) return;
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });

    const progressBar = document.getElementById('progress-bar');
    const updateProgress = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const pct = max > 0 ? window.scrollY / max * 100 : 0;
      progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    };
    addEventListener('scroll', updateProgress, { passive: true });
    addEventListener('resize', updateProgress);
    updateProgress();

    const headings = [...document.querySelectorAll('.article h2')];
    const links = [...document.querySelectorAll('.sidebar .outline-links a')];
    if (headings.length && links.length) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (!visible) return;
        const id = visible.target.id;
        links.forEach(link => link.classList.toggle('active', link.hash.slice(1) === id));
      }, { rootMargin: '-12% 0px -72% 0px', threshold: 0 });
      headings.forEach(heading => observer.observe(heading));
    }
"""


if __name__ == "__main__":
    OUT.write_text(build_page(), encoding="utf-8")
    print(OUT.resolve())
