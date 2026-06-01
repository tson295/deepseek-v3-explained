# FAQ — Câu hỏi thường gặp khi báo cáo DeepSeek-V3

> Bộ câu hỏi–trả lời để chuẩn bị khi thuyết trình/bảo vệ báo cáo. Sắp xếp từ **câu hỏi tổng quan** → **câu hỏi kỹ thuật sâu** → **câu hỏi "bẫy"/phản biện**.
>
> Mỗi câu trả lời có: **đáp án ngắn** (nói khi bị hỏi nhanh) + **giải thích** (nếu được hỏi sâu thêm).

---

## A. CÂU HỎI TỔNG QUAN

### A1. DeepSeek-V3 là gì? Tóm tắt trong 30 giây.
**Ngắn:** Một LLM mã nguồn mở kiến trúc Mixture-of-Experts, 671 tỉ tham số nhưng chỉ kích hoạt 37 tỉ mỗi token, đạt hiệu năng ngang GPT-4o/Claude-3.5 nhưng chi phí huấn luyện chỉ ~5,6 triệu USD.

**Giải thích:** Điểm cốt lõi là **đồng thiết kế thuật toán–framework–phần cứng**. Họ không chỉ làm model mạnh, mà làm model mạnh **với chi phí thấp bất ngờ**, nhờ một loạt đổi mới: MLA (attention tiết kiệm bộ nhớ), cân bằng tải không cần hàm phụ, MTP, huấn luyện FP8, DualPipe, và chưng cất từ DeepSeek-R1.

### A2. Điểm mới quan trọng nhất là gì?
**Ngắn:** Nếu chọn 1, đó là **huấn luyện FP8 ở quy mô siêu lớn** — lần đầu được chứng minh khả thi cho mô hình hàng trăm tỉ tham số. Nếu chọn về tư duy, đó là **co-design** toàn hệ thống.

**Giải thích:** Mỗi đổi mới phá vỡ một đánh đổi cố hữu (xem bảng cuối). FP8 đáng kể nhất vì nó vừa khó (dễ phân kỳ), vừa tác động trực tiếp tới chi phí (nhanh 2×, bộ nhớ giảm nửa).

### A3. Tại sao bài báo này gây tiếng vang lớn?
**Ngắn:** Vì nó phá vỡ định kiến "muốn model frontier phải đốt hàng trăm triệu đô". Một lab chứng minh có thể đạt đẳng cấp GPT-4o với chi phí thấp hơn nhiều bậc, và **công khai cách làm**.

### A4. DeepSeek-V3 khác DeepSeek-R1 thế nào?
**Ngắn:** V3 là mô hình **nền tảng đa năng** (chat, code, toán). R1 là mô hình **chuyên suy luận (reasoning)** dạng long Chain-of-Thought. V3 **học (distill) năng lực suy luận từ R1** trong giai đoạn hậu huấn luyện.

**Giải thích:** R1 suy nghĩ rất sâu nhưng dài dòng, định dạng kém. V3 chắt lọc cái hay của R1 (độ chính xác, pattern phản tỉnh/kiểm chứng) mà vẫn giữ phản hồi gọn gàng.

---

## B. CÂU HỎI VỀ KIẾN TRÚC

### B1. MLA hoạt động thế nào và tại sao tốt hơn MHA?
**Ngắn:** MLA nén key/value xuống một vector tiềm ẩn nhỏ (512 chiều thay vì 16384), chỉ cache vector nén đó. KV cache giảm ~57 lần mà chất lượng vẫn ngang MHA đầy đủ.

**Giải thích:** MHA cache K,V đầy đủ → tốn bộ nhớ khủng (512 GB cho context 128K!). MQA/GQA giảm bộ nhớ nhưng hi sinh chất lượng. MLA dùng nén hạng thấp → vừa nhỏ (9 GB) vừa giữ chất lượng. Chi tiết: phải **tách riêng nhánh RoPE** vì phép quay theo vị trí không tương thích với nén.

### B2. "Cân bằng tải không cần hàm phụ" — cũ làm sao, mới làm sao?
**Ngắn:** Cũ dùng hàm phạt (auxiliary loss) ép router chia đều token, nhưng gradient của nó xung đột với mục tiêu ngôn ngữ → hại chất lượng. Mới dùng một **bias term** chỉ ảnh hưởng việc *chọn* expert, không qua gradient, nên không hại chất lượng.

**Giải thích:** Bias được điều chỉnh bằng quan sát tải: expert quá tải thì giảm bias, thiếu tải thì tăng. Giá trị nhân vào output vẫn là điểm gốc. Tách hoàn toàn "cân bằng tải" khỏi "học ngôn ngữ".

### B3. Routing collapse là gì?
**Ngắn:** Hiện tượng router dồn hầu hết token vào một số ít expert "ưa thích", các expert khác bị bỏ đói. Gây mất cân bằng tải → lãng phí GPU trong expert parallelism.

### B4. MTP là gì? Có làm chậm suy luận không?
**Ngắn:** Multi-Token Prediction — huấn luyện mô hình dự đoán nhiều token tương lai cùng lúc. **Không** làm chậm suy luận: lúc deploy có thể vứt module MTP đi (chi phí y hệt baseline), hoặc dùng nó để **tăng tốc** qua speculative decoding (TPS × 1.8).

**Giải thích:** Lợi ích lúc train: tín hiệu giám sát dày hơn, mô hình học "lập kế hoạch" biểu diễn. Ablation chứng minh nó cải thiện hầu hết benchmark.

### B5. Shared expert khác routed expert thế nào?
**Ngắn:** Shared expert **luôn** kích hoạt cho mọi token (học kiến thức chung). Routed expert chỉ kích hoạt khi router chọn (học kiến thức chuyên biệt). V3: 1 shared + 256 routed, mỗi token dùng 8 routed.

---

## C. CÂU HỎI VỀ HUẤN LUYỆN (FP8, hạ tầng)

### C1. FP8 là gì và tại sao trước đây khó dùng cho pre-training?
**Ngắn:** FP8 = số thực 8 bit (nhanh 2×, tốn nửa bộ nhớ so với BF16). Khó vì dải biểu diễn hẹp + nhạy với outlier: một giá trị ngoại lai làm chết độ chính xác cả tensor → dễ phân kỳ.

**Giải thích:** Trước đây FP8 chỉ thành công ở inference (dễ hơn). DeepSeek là **lần đầu** chứng minh nó dùng được cho pre-training quy mô hàng trăm tỉ tham số.

### C2. DeepSeek giải quyết bất ổn FP8 bằng cách nào?
**Ngắn:** 4 kỹ thuật: (1) **lượng tử hóa chi tiết** theo nhóm nhỏ (1×128, 128×128) để cô lập outlier; (2) **promotion lên CUDA Core FP32** để tích lũy chính xác (vá lỗi 14-bit của H800); (3) dùng **E4M3** cho mọi tensor; (4) **online quantization**.

**Giải thích sâu (1):** scale per-tensor để một outlier kéo căng cả thang đo; chia nhóm nhỏ thì outlier chỉ hại nhóm của nó. **(2):** Tensor Core H800 cộng dồn FP8 chỉ giữ ~14 bit, sai số tới ~2% khi K lớn; họ chuyển sang FP32 mỗi 128 phần tử.

### C3. Có ví dụ cụ thể về việc FP8 phân kỳ không?
**Ngắn:** Có — Phụ lục A.2. Khi họ thử lượng tử hóa block-wise 128×128 cho **activation gradient**, mô hình 16B **phân kỳ** sau ~300B token.

**Giải thích:** Activation gradient có "token-correlated outliers" cực mất cân bằng. Phép Dgrad lan ngược theo chuỗi nên rất nhạy. Block quá thô → phá gradient → diverge. Đây là lý do họ buộc phải dùng tile 1×128 chi tiết hơn cho activation. Cho thấy ranh giới thành công/phân kỳ rất mỏng.

### C4. DualPipe là gì?
**Ngắn:** Một thuật toán pipeline parallelism hai chiều, **chồng lấp tính toán với giao tiếp** để giấu overhead, đồng thời giảm "pipeline bubble" (thời gian GPU ngồi chờ).

**Giải thích:** MoE xuyên node có giao tiếp all-to-all nặng (~1:1 với tính toán). DualPipe giấu giao tiếp này sau tính toán → khi scale lớn, overhead giao tiếp vẫn ~0. Đánh đổi: giữ 2 bản sao tham số (nhưng không đáng kể vì EP size lớn).

### C5. Tại sao họ không dùng Tensor Parallelism?
**Ngắn:** Vì tối ưu bộ nhớ tốt (recompute, EMA trên CPU, chia sẻ tham số MTP) nên không cần TP. TP có chi phí giao tiếp cao; tránh được thì tiết kiệm.

### C6. Tại sao họ viết hẳn "đề xuất thiết kế phần cứng"?
**Ngắn:** Vì nhiều thủ thuật của họ là để **lách thiếu sót phần cứng hiện tại** (vd lỗi 14-bit accumulation, thiếu hỗ trợ fine-grained quantization). Họ đề xuất chip tương lai khắc phục — nhiều đề xuất sau này có trên kiến trúc Blackwell.

---

## D. CÂU HỎI VỀ HẬU HUẤN LUYỆN

### D1. GRPO khác PPO thế nào?
**Ngắn:** PPO cần một critic model to bằng policy → tốn gấp đôi. GRPO **bỏ critic**, dùng điểm trung bình của một nhóm output làm baseline. Tiết kiệm tài nguyên lớn.

**Giải thích:** Với mỗi câu hỏi, sample G output, chấm điểm, advantage = (điểm của output − trung bình nhóm)/độ lệch chuẩn. "Tốt hơn anh em cùng nhóm thì được thưởng."

### D2. Distillation từ R1 nghĩa là gì?
**Ngắn:** Dùng R1 (mô hình suy luận mạnh) sinh dữ liệu huấn luyện cho V3, để V3 học được năng lực suy luận long-CoT mà không cần tự "nghĩ dài".

**Giải thích:** Vấn đề: R1 dài dòng, định dạng kém. Giải pháp: xây expert model qua SFT+RL, sinh 2 loại mẫu (gốc + R1 với system prompt hướng phản tỉnh/kiểm chứng), rồi rejection sampling chắt lọc. Kết quả: MATH-500 74.6→83.2, nhưng độ dài phản hồi tăng (phải cân bằng).

### D3. Rule-based vs model-based reward?
**Ngắn:** Rule-based: bài có đáp án xác định (toán đóng khung, code chạy test) → chống gian lận tuyệt đối. Model-based: bài đáp án tự do → dùng reward model (huấn luyện từ checkpoint V3, kèm chain-of-thought để giảm reward hacking).

### D4. "Self-rewarding" là gì?
**Ngắn:** V3 đủ giỏi để **tự làm giám khảo** cho chính mình (trên RewardBench đạt 89.6 với voting, cao nhất). Họ dùng chính V3 + voting làm nguồn feedback cho alignment ở các bài không có đáp án rõ ràng (theo hướng constitutional AI).

---

## E. CÂU HỎI VỀ KẾT QUẢ

### E1. DeepSeek-V3 mạnh nhất ở đâu, yếu nhất ở đâu?
**Ngắn:** Mạnh nhất: **toán** (MATH-500 90.2, AIME 39.2 — bỏ xa mọi đối thủ) và **code thuật toán** (Codeforces, LiveCodeBench dẫn đầu). Yếu hơn: **factuality tiếng Anh** (SimpleQA 24.9 thua GPT-4o 38.2), **code kỹ thuật** (SWE-Bench thua Claude), GPQA thua Claude.

**Giải thích:** Toán mạnh nhờ distill R1. SimpleQA tiếng Anh yếu vì họ ưu tiên token cho kiến thức tiếng Trung (đổi lại C-SimpleQA dẫn đầu 64.8).

### E2. So với LLaMA-3.1-405B thì sao? V3 có 671B mà?
**Ngắn:** V3 có 671B **tổng** nhưng chỉ **37B kích hoạt**/token — ít hơn 405B của LLaMA tới 11 lần về tham số kích hoạt. Vậy mà V3 vẫn vượt LLaMA-3.1-405B trên đa số benchmark. Đây là sức mạnh của MoE thưa.

### E3. "Ngang GPT-4o/Claude-3.5" — có phóng đại không?
**Ngắn:** Trên benchmark thì có cơ sở (xem bảng), đặc biệt toán/code/open-ended. Nhưng cần thận trọng: benchmark không phản ánh hết, và bản thân tác giả cũng cảnh báo nguy cơ "over-fit benchmark". Trên một số mặt (SWE-Bench, GPQA, SimpleQA EN) vẫn còn thua.

---

## F. CÂU HỎI "BẪY" / PHẢN BIỆN

### F1. Có thật DeepSeek làm model frontier chỉ với 5,6 triệu USD?
**Ngắn:** **Không hẳn.** Con số $5,576M **chỉ** là lần huấn luyện chính thức cuối cùng. Nó **KHÔNG** gồm chi phí R&D, ablation, thử nghiệm kiến trúc, xây dữ liệu, lương kỹ sư, hay khấu hao cụm GPU. Bài báo nói rõ điều này.

**Giải thích:** Đây là điểm hay bị truyền thông phóng đại. Tổng chi phí thực để ra được V3 (gồm cả R1, V2, mọi thử nghiệm) lớn hơn rất nhiều. Con số $5,6M chỉ chứng minh **lần train cuối hiệu quả**, không phải "toàn bộ chi phí dự án".

### F2. MoE 671B có thực sự "rẻ" khi triển khai không?
**Ngắn:** Train thì rẻ (chỉ tính 37B/token), nhưng **deploy thì không nhẹ**: phải chứa toàn bộ 671B tham số trong bộ nhớ. Đơn vị triển khai tối thiểu là 32 GPU (prefilling) đến 320 GPU (decoding) — gánh nặng cho nhóm nhỏ. Đây là **hạn chế tác giả tự nêu**.

### F3. Họ dùng output của GPT-4/Claude để train không? (vấn đề đạo đức/bản quyền)
**Ngắn:** Bài báo nói dữ liệu suy luận sinh từ **DeepSeek-R1** (mô hình của họ), non-reasoning từ **DeepSeek-V2.5**. Không tuyên bố dùng output mô hình closed-source. Tuy nhiên đây là vấn đề thường gây tranh luận với các model Trung Quốc — nên trả lời "theo bài báo thì..." và không khẳng định quá.

### F4. FP8 dưới 0,25% sai số — chứng minh trên mô hình bao lớn?
**Ngắn:** Trên mô hình ~16B và ~230B, train ~1T token (không phải trên chính bản 671B đầy đủ vì quá tốn). Đây là điểm có thể bị chất vấn: bằng chứng là extrapolation từ quy mô nhỏ hơn, dù chính V3 671B cũng train ổn định bằng FP8.

### F5. Đổi mới nào thực sự "của họ", cái nào kế thừa?
**Ngắn:** Kế thừa/dựa trên công trình khác: MLA, DeepSeekMoE (từ V2), MTP (cảm hứng từ Meta), GRPO (từ DeepSeekMath), YaRN, bias-free balancing (cảm hứng từ noaux_tc). **Đóng góp mới của V3:** tích hợp + scale tất cả lên 671B, **FP8 framework ở quy mô siêu lớn**, **DualPipe**, **kernel all-to-all**, **pipeline distill từ R1**. Sức mạnh nằm ở **tích hợp & co-design**, không phải một thuật toán đơn lẻ.

### F6. Nếu hỏi "hạn chế lớn nhất của bài báo này"?
**Ngắn:** (1) Deploy nặng (cần nhiều GPU). (2) Phụ thuộc thủ thuật lách phần cứng hiện tại. (3) Bằng chứng FP8 dựa trên quy mô nhỏ hơn. (4) Một số tuyên bố "ngang frontier" cần thận trọng vì giới hạn của benchmark. (5) Con số chi phí dễ gây hiểu lầm.

---

## G. CÂU HỎI "TẠI SAO" SÂU (hiểu bản chất)

### G1. Tại sao tách RoPE ra khỏi nén MLA?
Vì RoPE áp phép quay phụ thuộc vị trí. Nếu nén chung, ma trận up-projection bị "vướng" phép quay khác nhau ở mỗi vị trí → không gộp (absorb) được → mất lợi ích nén. Tách nhánh nhỏ riêng cho RoPE giải quyết.

### G2. Tại sao bias term không gây xung đột gradient mà auxiliary loss thì có?
Vì bias **chỉ tham gia bước chọn top-K** (một thao tác rời rạc, không khả vi theo cách ảnh hưởng loss ngôn ngữ), và được cập nhật bằng **luật quan sát tải** chứ không qua backprop. Auxiliary loss thì là một số hạng trong tổng loss → gradient của nó trực tiếp trộn vào và kéo lùi mục tiêu chính.

### G3. Tại sao cân bằng theo batch tốt hơn theo sequence?
Cân bằng theo sequence ép **mỗi chuỗi** dùng đều mọi expert → cản expert chuyên môn hóa theo domain. Cân bằng theo batch lỏng hơn → cho phép một chuỗi "toán" dùng nhiều expert toán, một chuỗi "code" dùng nhiều expert code → **chuyên môn hóa tốt hơn**. Ablation xác nhận: expert specialization cao hơn, loss thấp hơn.

### G4. Tại sao MTP giúp mô hình chính mạnh hơn dù bị vứt lúc suy luận?
Vì trong lúc train, MTP buộc biểu diễn ẩn $\mathbf{h}_i$ phải chứa đủ thông tin để dự đoán **không chỉ token kế mà cả token sau nữa** → biểu diễn "giàu" và "có kế hoạch" hơn. Biểu diễn tốt hơn này nằm trong mô hình chính, nên dù vứt module MTP, mô hình chính vẫn hưởng lợi.

### G5. Tại sao decoding cần nhiều GPU hơn prefilling (320 vs 32)?
Decoding nghẽn ở **truy cập bộ nhớ** (memory-bound), batch nhỏ. Để mỗi expert có đủ token xử lý hiệu quả và giữ độ trễ thấp, họ trải expert ra cực rộng (EP320, mỗi GPU 1 expert). Prefilling là **compute-bound**, batch lớn nên gói gọn hơn.

---

## H. "CHỐT HẠ" — 3 ý cần nhớ nếu chỉ nói được 3 câu

1. **DeepSeek-V3 chứng minh "frontier với chi phí thấp" là khả thi** — nhờ đồng thiết kế thuật toán–framework–phần cứng, không phải một viên đạn bạc.
2. **Mỗi đổi mới phá vỡ một đánh đổi cố hữu** (MLA: bộ nhớ↔chất lượng; aux-free: cân bằng↔chất lượng; FP8: tốc độ↔ổn định; DualPipe: song song↔giao tiếp; GRPO: chất lượng↔chi phí critic).
3. **Cẩn thận với con số $5,6M** — đó là lần train cuối, không phải tổng chi phí dự án.
