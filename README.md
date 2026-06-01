# DeepSeek-V3, giải thích từ đầu đến cuối

Một trang web tĩnh giải thích kiến trúc và quy trình huấn luyện của **DeepSeek-V3**
theo kiểu giáo trình: mỗi kỹ thuật (MoE, MLA, FP8, DualPipe, GRPO, YaRN, …) đều có
bối cảnh, lý do xuất hiện, cách hoạt động, vì sao tốt hơn cách cũ và cái giá phải trả.

🔗 **Xem trực tiếp:** https://tson295.github.io/deepseek-v3-explained/

## Nội dung

- `index.html` — trang web đã render (giao diện docs, sáng/tối, mục lục cuộn theo, MathJax).
- `build_explained_html.py` — script Python sinh ra `index.html` từ nội dung Markdown nhúng trong file.
- `GIAI_THICH_CHI_TIET_DeepSeek-V3.md`, `BAO_CAO_DeepSeek-V3.md`, `FAQ_DeepSeek-V3.md` — tài liệu nguồn.

## Build lại trang

```bash
pip install markdown beautifulsoup4
python build_explained_html.py   # ghi đè index.html
```

Mọi nội dung và CSS/JS nằm trong `build_explained_html.py`; sửa ở đó rồi chạy lại
script để cập nhật `index.html`.

## Nguồn

Biên soạn lại từ DeepSeek-V3 Technical Report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
Công thức render bằng [MathJax](https://www.mathjax.org/).
