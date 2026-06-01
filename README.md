# DeepSeek-V3, giải thích từ đầu đến cuối

Một trang web tĩnh giải thích kiến trúc và quy trình huấn luyện của **DeepSeek-V3**
theo kiểu giáo trình: mỗi kỹ thuật (MoE, MLA, FP8, DualPipe, GRPO, YaRN, …) đều có
bối cảnh, lý do xuất hiện, cách hoạt động, vì sao tốt hơn cách cũ và cái giá phải trả.

Mỗi phần chính kèm một **sơ đồ minh hoạ vẽ bằng SVG** (12 sơ đồ) để nhìn là hiểu ý
tưởng trước khi đọc công thức — sơ đồ tự đổi màu theo giao diện sáng/tối.

🔗 **Xem trực tiếp:** https://tson295.github.io/deepseek-v3-explained/

## Nội dung

- `index.html` — trang web đã render (giao diện docs, sáng/tối, mục lục cuộn theo, MathJax). **Tự sinh ra — không sửa tay.**
- `build_explained_html.py` — script Python sinh ra `index.html` từ nội dung Markdown nhúng trong file. **Đây là nơi cần sửa.**
- `requirements.txt` — dependencies để build (markdown, beautifulsoup4).
- `.github/workflows/build.yml` — CI tự build lại `index.html` mỗi khi push lên `main`.
- `GIAI_THICH_CHI_TIET_DeepSeek-V3.md`, `BAO_CAO_DeepSeek-V3.md`, `FAQ_DeepSeek-V3.md` — tài liệu nguồn.

## Tự động build (CI)

Quy trình giờ tự động: **sửa `build_explained_html.py` (hoặc các `.md`) rồi push lên
`main` là xong.** GitHub Actions (`.github/workflows/build.yml`) sẽ chạy lại script,
build lại `index.html` và commit ngược lên `main`; GitHub Pages tự phát hành bản mới.
Không cần build tay, cũng không cần đổi cài đặt nào của repo. Pull request thì CI chỉ
kiểm tra và báo lỗi nếu `index.html` bị quên build lại.

## Build lại trang (thủ công, nếu cần)

```bash
pip install -r requirements.txt
python build_explained_html.py   # ghi đè index.html
```

Mọi nội dung và CSS/JS nằm trong `build_explained_html.py`; sửa ở đó rồi chạy lại
script để cập nhật `index.html`.

### Sơ đồ minh hoạ

Các sơ đồ là SVG vẽ tay, định nghĩa trong `dict` `DIAGRAMS` và chèn vào bài qua
placeholder chữ HOA (ví dụ `DIAGRAMSLOTMLAFLOW`) đứng riêng một dòng trong
`ARTICLE_MD`. Việc thay placeholder bằng SVG diễn ra **sau** bước BeautifulSoup
(xem `build_page`) vì parser HTML hạ thấp các thuộc tính camelCase như `viewBox`,
làm hỏng SVG. Màu lấy từ biến CSS (`--accent`, `--d-warn`, …) nên sơ đồ tự khớp
giao diện sáng/tối; thêm/sửa sơ đồ chỉ cần đụng `DIAGRAMS` và đặt placeholder.

## Nguồn

Biên soạn lại từ DeepSeek-V3 Technical Report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
Công thức render bằng [MathJax](https://www.mathjax.org/).
