<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
    🎓 Khoa Công nghệ thông tin (Đại học Đại Nam)
    </a>
</h2>
<h2 align="center">
  Phát hiện và phân loại ngôn ngữ độc hại trong văn bản Tiếng Việt
</h2>
<div align="center">
    <p align="center">
        <img width="200" alt="dnu_logo" src="https://github.com/user-attachments/assets/2bcb1a6c-774c-4e7d-b14d-8c53dbb4067f" />
        <img width="180" alt="fitdnu_logo" src="https://github.com/user-attachments/assets/ec4815af-e477-480b-b9fa-c490b74772b8" />
        <img width="170" alt="aiotlab_logo" src="https://github.com/user-attachments/assets/41ef702b-3d6e-4ac4-beac-d8c9a874bca9" />
    </p>

[![DaiNam University](https://img.shields.io/badge/DaiNam_University-ff6b35?style=flat-square)](https://dainam.edu.vn)
[![Faculty of IT](https://img.shields.io/badge/Faculty_of_IT-0066cc?style=flat-square)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![AIoTLab](https://img.shields.io/badge/AIoTLab-28a745?style=flat-square&logo=facebook)](https://www.facebook.com/DNUAIoTLab)

</div>

## Mục lục
I. [Giới thiệu hệ thống](#giới-thiệu-hệ-thống)

II. [Mục tiêu](#mục-tiêu)

III. [Công nghệ sử dụng](#công-nghệ-sử-dụng)

IV. [Tính năng / Chức năng của hệ thống](#tính-năng--chức-năng-của-hệ-thống)

V. [Cấu trúc dự án](#cấu-trúc-dự-án)

VI. [Sơ đồ kiến trúc](#sơ-đồ-kiến-trúc)

VII. [Cách cài đặt](#cách-cài-đặt)

VIII. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

IX. [Hướng mở rộng](#hướng-mở-rộng)

---

## I. Giới thiệu hệ thống

Đây là hệ thống **web demo** dùng để:
- Phát hiện xem **một đoạn văn tiếng Việt** có chứa **ngôn ngữ độc hại / xúc phạm / tiêu cực** hay không.
- **Tô sáng (highlight)** đúng những **từ/cụm từ** bị coi là độc hại để người duyệt/bộ phận quản lý nội dung xem rất nhanh.
- Hỗ trợ cả **kiểm tra từng đoạn** và **kiểm tra theo lô (upload file)**, phù hợp cho bài tập, đồ án, hoặc mô-đun kiểm duyệt nội dung nội bộ.

Hệ thống được thiết kế **theo hướng dễ giải thích** (explainable): không chỉ trả về “câu này bẩn”, mà còn chỉ rõ **vì sao bẩn** (vị trí, từ điển nào khớp, mô hình đoán ra hay do viết tắt).

Màu chủ đạo: **xanh – cam – trắng** theo đúng UI bạn đang dùng:  
- Xanh: thanh header, nút chính  
- Cam: tiêu đề box (“Phân tích 1 đoạn”, “Phân tích file…”, “Kết quả”)  
- Trắng: nền card, phần văn bản highlight màu vàng nhạt (không gạch chân)

---


## II. Mục tiêu
- **Phát hiện tự động** các đoạn phản hồi/sinh viên/bình luận có ngôn từ độc hại.
- **Highlight chính xác** các **span** từ/cụm từ vi phạm để người dùng nhận biết ngay.
- Hỗ trợ **ngưỡng cảnh báo** để tuỳ ý siết/chùng (ví dụ 50%, 70%…).
- Hỗ trợ **phân tích hàng loạt** (tối đa 200 dòng) để giáo viên/quản trị có thể rà file góp ý lớn.
- Trả kết quả theo **định dạng chuẩn** (CSV / DOCX) để đính kèm báo cáo hoặc nộp môn.
- Làm mẫu **đề tài “phát hiện và phân loại ngôn ngữ độc hại trong văn bản tiếng Việt”** ở mức có thể demo, trình bày, và mở rộng.

---

## III. Công nghệ sử dụng
- **Backend**: `Python` + `Flask`
  - REST API: `/api/predict`, `/api/upload`, `/api/export_docx`
  - Xử lý file: CSV, TXT, (có thể mở rộng XLSX/DOCX/PDF)
- **Xử lý/ngôn ngữ**:
  - Tiền xử lý tiếng Việt đơn giản: lower, bỏ khoảng trắng thừa
  - **Từ điển (lexicon) + regex** ngôn từ xúc phạm / viết tắt phổ biến (`dm`, `đm`, `vcl`, `thối lợm`, …)
  - **ML nhẹ**: TF-IDF (1–2 gram) + Logistic Regression → cho **xác suất xúc phạm**
  - **Hybrid** = (prob ≥ ngưỡng) **hoặc** (có từ trong lexicon) → phù hợp cho web real-time
- **Frontend**: `HTML5`, `CSS3`, `JavaScript` (file tách riêng `index.html`, `style.css`, `app.js`)
  - UI theo thẻ/card
  - **Chart.js** (hoặc canvas tương đương) để vẽ **biểu đồ doughnut**: cam = xúc phạm, xanh = không
- **Tài liệu/ảnh**: SVG sơ đồ kiến trúc trong thư mục `docs/`

---

## IV. Chức năng của hệ thống

### 1. Phân tích 1 đoạn
- Ô nhập văn bản (“Đoạn phản hồi”)
- Nút **“Phân tích”** → gọi API `/api/predict`
- Nút **“Xóa”** → xoá nội dung + kết quả
- Kết quả:
  - Kết luận: **Xúc phạm / Không**
  - Xác suất (%)
  - Danh sách **spans** (từ/cụm từ vi phạm) kèm vị trí
  - Đoạn văn đã **highlight** (`<mark>` màu vàng nhạt)
  - **Biểu đồ doughnut** (cam/xanh)

### 2. Phân tích file (batch)
- Chọn tệp: **CSV / TXT / XLSX / DOCX / PDF**
- Nút **“Tải lên & phân tích”**
- Hệ thống đọc tối đa **200 dòng**
- Hiển thị:
  - Bảng tổng hợp: STT, Xác suất, Kết luận, Văn bản
  - Biểu đồ doughnut tổng (tỷ lệ xúc phạm / không)
  - Có thể xuất CSV / DOCX

### 3. Ngưỡng cảnh báo
- Slider (ví dụ mặc định 50%)
- Rule: **kết luận = xúc phạm** nếu  
  - `probability ≥ threshold` **hoặc**
  - Có ít nhất 1 span từ từ điển / viết tắt
- Dùng để test trong demo: kéo slider và chạy lại để xem kết quả đổi

### 4. Redact (ẩn từ bẩn)
- Checkbox / toggle: “Ẩn phần bị gắn cờ (***, Redact)”
- Nếu bật: những đoạn bị highlight sẽ được thay bằng `***` khi hiển thị
- Phù hợp khi đưa ra màn hình lớn/trước lớp

### 5. Biểu đồ trực quan
- Dạng **doughnut** nhỏ gọn
- Màu:
  - **Cam**: xúc phạm
  - **Xanh**: không
- Ở mode batch: biểu đồ tổng
- Ở mode đơn lẻ: biểu đồ theo 1 câu

### 6. Xuất báo cáo
- Backend đã chuẩn bị logic để xuất tài liệu có highlight
- Có thể mở rộng thêm:
  - CSV (client)
  - DOCX (server)
  - JSONL (gắn nhãn)

---

## V. Cấu trúc dự án

```text
.
├─ app.py                  # Flask app, route API, load model, xử lý file
├─ requirements.txt        # Thư viện Python cần cài
├─ templates/
│  └─ index.html           # Giao diện chính (form phân tích 1 đoạn, form upload)
├─ static/
│  ├─ style.css            # Toàn bộ CSS: header xanh, box cam, layout 2 cột
│  ├─ app.js               # JS gọi API, vẽ biểu đồ, render kết quả
│  └─ dainam_logo.png      # (nếu bạn thêm logo)
├─ data/
│  ├─ data_train.csv       # (tuỳ chọn) dữ liệu huấn luyện bạn có
│  └─ data_eval.csv        # dữ liệu đánh giá mẫu để chạy script
├─ scripts/
│  └─ eval_offensive.py    # script đánh giá P/R/F1/AUC cho 3 mô hình: Lexicon, ML, Hybrid
├─ docs/
│  ├─ architecture_dainam.svg          # sơ đồ kiến trúc
│  ├─ screenshot-home-exact.svg        # giao diện trang chính
│  ├─ screenshot-batch-exact.svg       # giao diện phân tích file
│  └─ screenshot-guide-exact.svg       # giao diện hướng dẫn sử dụng
└─ README.md               # (file bạn đang đọc) 

```

## 6. Sơ đồ kiến trúc

Hệ thống gồm 4 lớp chính:

1. **Giao diện người dùng (Frontend)**
   - File: `templates/index.html`, `static/style.css`, `static/app.js`
   - Nhiệm vụ:
     - Cho phép **nhập 1 đoạn** hoặc **tải lên file** (CSV/TXT/XLSX/DOCX/PDF)
     - Gửi request tới API Flask
     - Hiển thị lại **đoạn văn đã highlight**, **bảng kết quả**, **biểu đồ doughnut**
     - Cho phép người dùng chỉnh **ngưỡng cảnh báo** và **bật tắt Redact**

2. **API & Điều phối (Flask Backend)**
   - File: `app.py`
   - Endpoint chính:
     - `POST /api/predict` – phân tích 1 đoạn
     - `POST /api/upload` – phân tích nhiều dòng (tối đa 200)
     - `POST /api/export_docx` – xuất báo cáo có highlight
   - Đảm nhiệm:
     - Nhận dữ liệu từ client
     - Gọi pipeline xử lý tiếng Việt
     - Đóng gói dữ liệu trả về đúng format cho UI

3. **Tầng xử lý & mô hình (Detection Engine)**
   - Các bước:
     1. **Tiền xử lý**: hạ chữ, chuẩn hoá viết tắt/slang (`dm`, `đm`, `vcl`, …)
     2. **Dò từ điển (lexicon + regex)**: tìm nhanh các từ/cụm xúc phạm → sinh **span**
     3. **Mô hình ML nhẹ**: TF-IDF (1–2 gram) + Logistic Regression → sinh **xác suất xúc phạm**
     4. **Gộp span**: hợp nhất span lexicon + span từ ML để tránh chồng lấn
     5. **Luật quyết định (hybrid)**:
        - Nếu `probability >= threshold` **hoặc** có ít nhất 1 span từ từ điển → **gắn nhãn Xúc phạm**
   - Điểm mạnh: nhanh, dễ giải thích, phù hợp web real-time

4. **Dữ liệu & Báo cáo**
   - Thư mục: `data/` (ví dụ: `data_train.csv`, `data_eval.csv`)
   - Dùng để:
     - Huấn luyện / thử nghiệm mô hình cục bộ
     - Xuất kết quả dạng CSV/DOCX để nộp báo cáo
   - Có thể bổ sung bộ từ điển riêng theo từng khoa/lớp

> `![Sơ đồ kiến trúc](docs/architecture_dainam.svg)`

---

## VII. Cách cài đặt

### 1. Yêu cầu
- Python **3.9+**
- pip
- Khuyến nghị: môi trường ảo `venv`

### 2. Các bước cài đặt

```bash
# 1) Clone dự án
git clone <link-github-cua-ban>
cd <Chuyen_doi_so>
cd <Data>

# 2) Tạo môi trường ảo
python -m venv venv

# 3) Kích hoạt môi trường ảo
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# hoặc cmd:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 4) Cài thư viện
pip install -r requirements.txt

# 5) Chạy ứng dụng
python app.py
# mở trình duyệt: http://127.0.0.1:5000

```

## VIII. Hướng dẫn sử dụng

### A. Phân tích 1 đoạn
1. Vào trang chính.
2. Nhập đoạn văn vào ô “Phân tích 1 đoạn”.
3. Bấm nút “Phân tích”.
4. Xem kết quả ở box “Kết quả”:
   - Kết luận: Xúc phạm / Không
   - Xác suất (%)
   - Đoạn văn đã highlight (màu vàng nhạt, không gạch chân)
   - Biểu đồ doughnut (Cam = Xúc phạm, Xanh = Không)
5. Muốn làm lại → bấm “Xóa”.

### B. Phân tích file (tối đa 200 dòng)
1. Ở box “Phân tích file (CSV / TXT / XLSX / DOCX / PDF)” → bấm chọn tệp.
2. Chọn file góp ý / phản hồi của sinh viên.
3. Bấm “Tải lên & phân tích”.
4. Hệ thống sẽ:
   - Đọc tối đa 200 dòng
   - Phân tích từng dòng
   - Hiển thị bảng: STT, xác suất, kết luận, văn bản
   - Vẽ biểu đồ doughnut tổng
5. Dùng để test phần upload trong đề tài.


### C. Điều chỉnh ngưỡng cảnh báo
- Tại khung “Ngưỡng cảnh báo: 50%” → kéo thanh trượt.
- Quy tắc:
  + Nếu xác suất >= ngưỡng → Xúc phạm
  + Nếu có span từ điển → vẫn Xúc phạm dù xác suất < ngưỡng
- Dùng ngưỡng cao nếu muốn ít cảnh báo giả.

### D. Bật/tắt Redact
- Gạt nút “Ẩn phần bị gắn cờ (***, Redact)”.
- Khi bật: phần độc hại sẽ hiển thị dạng ***.
- Phù hợp khi demo trên lớp hoặc in báo cáo.

---

## IX. Hướng mở rộng
1. Phân loại đa lớp:
   - insult / profanity / hate / harassment / threat
   - UI hiển thị chip màu theo từng lớp

2. Quản lý từ điển ngay trên web (Lexicon Manager):
   - Thêm / xóa / sửa từ cấm
   - Import / Export dạng JSON
   - Lưu phiên bản theo thời gian

3. Mô hình nâng cao:
   - Thêm PhoBERT + token-classification để lấy span chính xác hơn
   - Vẫn giữ TF-IDF + Logistic làm fallback chạy nhanh

4. Evaluate trong giao diện:
   - Thêm 1 card “Đánh giá”
   - Upload file đã gán nhãn → tính Precision / Recall / F1 / AUC
   - Vẽ đường PR / ROC bằng Chart.js

5. Xuất nhiều định dạng:
   - CSV, DOCX, JSONL, HTML có `<mark>`
   - Tùy chọn ẩn thông tin nhạy cảm (PII-redaction) trước khi export

6. Triển khai thực tế:
   - Viết Dockerfile
   - Chạy bằng Gunicorn / Nginx
   - Thêm xác thực nếu dùng nội bộ

---

<div align="center">

## 👨‍💻 Tác giả

<img width="100" height="100" src="https://www.facebook.com/photo/?fbid=776760888610810&set=a.102069779413261" alt="Phạm Thị Huyền Trang" style="border-radius: 50%;">

**Phạm Thị Huyền Trang**  
🎓 CNTT 16-05  
🏛️ Khoa Công nghệ rhoong tin - Trường Đại học Đại Nam   

[![Email](https://img.shields.io/badge/Email-bbikemcutie@gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bbikemcutie@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-bbikem-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bbikem)

---

<sub> ❤️ [Phạm Thị Huyền Trang](https://www.facebook.com/hichanzz/) ❤️ </sub>

</div>
