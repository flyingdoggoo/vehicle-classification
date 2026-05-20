# BÁO CÁO ĐỒ ÁN KHOA HỌC DỮ LIỆU

## Đề tài

# PHÂN LOẠI PHƯƠNG TIỆN GIAO THÔNG BẰNG MÔ HÌNH RESNET VÀ MOBILENET

**Học phần:** Khoa học dữ liệu  
**Nhóm học phần:** 23Nh11  
**Năm thực hiện:** 2026  

| Họ tên sinh viên | Mã sinh viên | Nhóm HP | Vai trò chính |
|---|---:|---|---|
| Nguyễn Thanh Hiếu | 102230239 | 23Nh11 | Phụ trách chính phần AI, xây dựng notebook, MobileNet, đánh giá mô hình |
| Nguyễn Mạnh Kiên | 102230248 | 23Nh11 | Phụ trách chính phần AI, xây dựng ResNet, xử lý pipeline train/evaluate |
| Nguyễn Văn Tiến | 102230271 | 23Nh11 | Phụ trách chính crawl dữ liệu, kiểm tra nguồn ảnh, hỗ trợ thống kê dữ liệu |
| Huỳnh Ngọc Khánh Linh | 102230196 | 23Nh11 | Phụ trách chính crawl dữ liệu, tổ chức dữ liệu, hỗ trợ lọc thủ công |

**Ghi chú:** Cả nhóm cùng tham gia lọc dữ liệu thủ công, kiểm tra nhãn, tổng hợp kết quả và hoàn thiện báo cáo.

\newpage

# TÓM TẮT

Bài toán phân loại phương tiện giao thông từ ảnh là một bài toán thị giác máy tính có tính ứng dụng cao trong quản lý giao thông, phân tích camera đô thị, hệ thống giám sát bến bãi, thống kê luồng phương tiện và xây dựng các hệ thống hỗ trợ giao thông thông minh. Trong đồ án này, nhóm xây dựng một pipeline hoàn chỉnh để thu thập dữ liệu ảnh từ Internet, làm sạch dữ liệu, lọc trùng, chia tập dữ liệu, tiền xử lý ảnh, huấn luyện và đánh giá hai mô hình học sâu là MobileNet và ResNet.

Tập dữ liệu của đề tài (phiên bản thực nghiệm cuối) gồm 9 nhãn chính: `bicycle`, `boat`, `bus`, `car`, `helicopter`, `minibus`, `motorcycle`, `train`, `truck` (đã loại `taxi` để giảm nhiễu nhãn). Dữ liệu được thu thập bằng quá trình crawl ảnh từ các nguồn tìm kiếm ảnh như Naver và DuckDuckGo, kết hợp công cụ AutoCrawler cho các nguồn được hỗ trợ. Trong quá trình thử nghiệm, Google Images có hiện tượng CAPTCHA nên không được chọn làm nguồn crawl chính trong pipeline cuối. Sau khi crawl, dữ liệu được chuẩn hóa cấu trúc thư mục theo từng nhãn, loại bỏ ảnh lỗi, loại bỏ ảnh quá nhỏ, chuẩn hóa định dạng JPEG RGB, đổi tên thống nhất, lọc ảnh trùng lặp bằng perceptual hashing và Hamming distance, sau đó chuẩn bị để upload lên Kaggle huấn luyện mô hình.

Về mô hình hóa, nhóm sử dụng hai kiến trúc CNN phổ biến là MobileNetV2 và ResNet50 bằng TensorFlow/Keras. MobileNetV2 được lựa chọn do sử dụng depthwise separable convolution, giúp giảm số lượng tham số và chi phí tính toán. ResNet50 được lựa chọn do có residual block và skip connection, giúp cải thiện khả năng lan truyền gradient trong mạng sâu và tăng năng lực biểu diễn đặc trưng. Hai mô hình được đánh giá bằng Accuracy, Precision, Recall, F1-score, confusion matrix, classification report, biểu đồ loss/accuracy và trực quan hóa các ảnh dự đoán đúng/sai trên test set.

Báo cáo này trình bày toàn bộ quy trình từ thu thập dữ liệu đến huấn luyện và đánh giá mô hình. Các hình minh họa được đánh mã theo từng nhóm nội dung để đối chiếu trực tiếp với notebook, log crawl, biểu đồ thống kê và kết quả đánh giá mô hình.

\newpage

# BẢNG PHÂN CÔNG NHIỆM VỤ

| Thành viên | Nhiệm vụ chính | Nhiệm vụ chi tiết | Mức độ hoàn thành |
|---|---|---|---|
| Nguyễn Thanh Hiếu | AI và notebook | Thiết kế pipeline notebook, xây dựng MobileNetV2, cấu hình train, lưu best model, tổng hợp metric | Hoàn thành |
| Nguyễn Mạnh Kiên | AI và đánh giá mô hình | Xây dựng ResNet50, xử lý đánh giá test set, confusion matrix, classification report, so sánh mô hình | Hoàn thành |
| Nguyễn Văn Tiến | Thu thập dữ liệu | Tổ chức keyword, chạy crawl, theo dõi tiến độ, kiểm tra nguồn crawl, hỗ trợ làm sạch dữ liệu | Hoàn thành |
| Huỳnh Ngọc Khánh Linh | Thu thập và tổ chức dữ liệu | Hỗ trợ crawl, kiểm tra cấu trúc folder, thống kê dữ liệu, hỗ trợ chuẩn bị dữ liệu upload Kaggle | Hoàn thành |
| Cả nhóm | Lọc dữ liệu và báo cáo | Lọc thủ công ảnh sai nhãn, rà soát ảnh trùng, tổng hợp hình minh họa, viết và chỉnh sửa báo cáo | Hoàn thành |

\newpage

# MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)  
2. [Thu thập và mô tả dữ liệu](#2-thu-thập-và-mô-tả-dữ-liệu)  
3. [Trích xuất đặc trưng và xử lý dữ liệu](#3-trích-xuất-đặc-trưng-và-xử-lý-dữ-liệu)  
4. [Mô hình hóa dữ liệu](#4-mô-hình-hóa-dữ-liệu)  
5. [Kết luận](#5-kết-luận)  
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)  
7. [Phụ lục](#phụ-lục)  

\newpage

# 1. GIỚI THIỆU

## 1.1. Bối cảnh bài toán

Trong các hệ thống giao thông hiện đại, dữ liệu hình ảnh ngày càng được sử dụng rộng rãi. Camera giao thông, camera tại bãi đỗ xe, camera tại cổng trường, trạm thu phí, nhà ga, bến xe hoặc cảng biển đều tạo ra lượng ảnh và video lớn. Nếu toàn bộ dữ liệu này phải được kiểm tra thủ công, chi phí nhân lực sẽ rất cao, tốc độ xử lý chậm và độ ổn định phụ thuộc nhiều vào người quan sát. Vì vậy, các hệ thống tự động nhận dạng và phân loại phương tiện giao thông có ý nghĩa thực tiễn rõ ràng.

Bài toán của nhóm là phân loại một ảnh đầu vào thành một trong các loại phương tiện giao thông đã định nghĩa. Ảnh có thể chứa nhiều bối cảnh khác nhau như đường phố, bến tàu, nhà ga, bãi đỗ, không trung hoặc môi trường đô thị. Những khác biệt này làm bài toán khó hơn so với một tập dữ liệu được chụp trong điều kiện cố định. Ngoài ra, một số lớp có hình dạng khá giống nhau, ví dụ `car` và `minibus`, `bus` và `minibus`, hoặc `truck` và `bus` trong một số góc chụp. Điều này khiến mô hình cần học được đặc trưng hình dạng tổng quát thay vì chỉ ghi nhớ màu sắc hoặc bối cảnh.

Đồ án không sử dụng các tập dữ liệu có sẵn theo dạng tải trực tiếp, mà yêu cầu sinh viên tự thu thập dữ liệu bằng cách crawl ảnh. Đây là một yêu cầu quan trọng vì chất lượng dữ liệu ảnh thu thập từ Internet thường không đồng đều. Ảnh có thể bị lỗi, kích thước quá nhỏ, trùng lặp, có watermark, có nhiều vật thể khác nhau hoặc sai nhãn. Vì vậy, phần thu thập và làm sạch dữ liệu không chỉ là bước phụ trợ mà là một phần trung tâm của đồ án.

![FIG-CRAWL-01: Màn hình chạy pipeline crawl dữ liệu](PLACEHOLDER_FIG_CRAWL_01)

## 1.2. Mục tiêu của đề tài

Đề tài hướng đến việc xây dựng một pipeline hoàn chỉnh cho bài toán phân loại ảnh phương tiện giao thông. Pipeline này bao gồm các giai đoạn chính: thu thập dữ liệu, làm sạch dữ liệu, lọc trùng, thống kê và trực quan hóa dữ liệu, chia tập train/validation/test, tiền xử lý ảnh, trích xuất đặc trưng, huấn luyện hai mô hình học sâu và đánh giá kết quả.

Các mục tiêu cụ thể gồm:

- Thu thập hơn 10.000 ảnh sạch sau khi xử lý.
- Dữ liệu được chia theo 9 nhãn phương tiện giao thông.
- Mỗi nhãn có số lượng ảnh tương đối đủ, tránh mất cân bằng quá nghiêm trọng.
- Có báo cáo thống kê mô tả dữ liệu bằng bảng và biểu đồ.
- Có mô tả quy trình làm sạch dữ liệu và lọc trùng bằng Hamming distance.
- Huấn luyện và đánh giá hai mô hình MobileNetV2 và ResNet50.
- Lưu đầy đủ best model, final model, history, prediction output để phục vụ kiểm tra sau train.
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score và confusion matrix.
- So sánh MobileNet và ResNet bằng bảng kết quả và biểu đồ.

## 1.3. Danh sách lớp phân loại

Phiên bản thực nghiệm cuối của đề tài sử dụng 9 lớp:

| STT | Label | Mô tả |
|---:|---|---|
| 1 | `bicycle` | Xe đạp, bao gồm xe đạp đường phố, xe đạp thể thao, xe đạp địa hình |
| 2 | `boat` | Thuyền, tàu nhỏ, thuyền máy, thuyền du lịch |
| 3 | `bus` | Xe buýt, xe khách, xe buýt thành phố, xe buýt công cộng |
| 4 | `car` | Ô tô con, sedan, hatchback, SUV, xe gia đình |
| 5 | `helicopter` | Trực thăng dân sự, trực thăng cứu hộ, trực thăng vận tải |
| 6 | `minibus` | Xe buýt nhỏ, shuttle bus, passenger van |
| 7 | `motorcycle` | Xe máy, mô tô, scooter |
| 8 | `train` | Tàu hỏa, tàu điện, đầu tàu, toa tàu |
| 9 | `truck` | Xe tải, xe chở hàng, xe container, xe giao hàng |

Các lớp này được chọn vì vừa phổ biến trong giao thông, vừa có mức độ phân biệt hình dạng khác nhau. Một số lớp có ranh giới rõ ràng như `helicopter` hoặc `boat`, trong khi một số lớp dễ nhầm hơn như `car`, `minibus`, `bus` và `truck`. Nhờ vậy, bài toán đủ đa dạng để đánh giá khả năng học đặc trưng của mô hình.

## 1.4. Cấu hình thực nghiệm của đề tài

Trong phần mô hình hóa, nhóm dùng hai kiến trúc MobileNetV2 và ResNet50 để so sánh giữa hướng mô hình nhẹ và hướng mô hình sâu hơn. Cả hai mô hình đều thay classifier cuối bằng lớp phân loại 9 nhãn đúng với bộ dữ liệu của nhóm.

Notebook sử dụng quy trình huấn luyện 2 giai đoạn. Giai đoạn đầu freeze backbone và train classifier head để mô hình thích nghi với 9 lớp phương tiện. Giai đoạn sau mở một phần các lớp cuối của backbone với learning rate nhỏ để tối ưu đặc trưng cho dữ liệu crawl của nhóm. Cách làm này giúp quá trình train ổn định hơn trên bộ dữ liệu khoảng 10.000 ảnh, đồng thời vẫn giữ khả năng so sánh giữa MobileNetV2 và ResNet50.

## 1.5. Tổng quan pipeline dự án

Pipeline tổng thể của nhóm gồm các bước:

1. Tạo danh sách label và keyword crawl.
2. Crawl ảnh từ Naver và DuckDuckGo, có hỗ trợ AutoCrawler.
3. Lưu ảnh thô vào `data/raw/<label>/`.
4. Loại bỏ ảnh lỗi, ảnh quá nhỏ, ảnh không mở được.
5. Chuẩn hóa ảnh về RGB JPEG và đổi tên thống nhất.
6. Lọc ảnh trùng bằng perceptual hashing.
7. Lọc thủ công các ảnh sai nhãn hoặc chất lượng kém.
8. Thống kê dữ liệu bằng CSV và biểu đồ.
9. Upload `data/cleaned` lên Kaggle.
10. Notebook tự chia train/validation/test theo tỉ lệ 70/15/15.
11. Tiền xử lý ảnh: resize 224x224, normalize [0,1], augmentation.
12. Train MobileNet và ResNet từ đầu.
13. Đánh giá, trực quan hóa dự đoán đúng/sai và lưu artifact.

![FIG-FLOW-01: Flow tổng quan dự án từ crawl đến evaluate](PLACEHOLDER_FIG_FLOW_01)

\newpage

# 2. THU THẬP VÀ MÔ TẢ DỮ LIỆU

## 2.1. Cấu trúc dữ liệu của project

Dữ liệu được tổ chức theo cấu trúc thư mục chuẩn của bài toán image classification:

```text
traffic_vehicle_classification/
├── data/
│   ├── raw/
│   │   ├── bicycle/
│   │   ├── boat/
│   │   ├── bus/
│   │   ├── car/
│   │   ├── helicopter/
│   │   ├── minibus/
│   │   ├── motorcycle/
│   │   ├── train/
│   │   └── truck/
│   ├── cleaned/
│   ├── duplicates/
│   ├── rejected/
│   └── splits/
├── notebooks/
├── scripts/
├── configs/
└── reports/
```

Thư mục `data/raw` chứa ảnh tải trực tiếp từ Internet. Đây là dữ liệu chưa được xử lý, có thể chứa ảnh lỗi, ảnh trùng, ảnh sai nhãn hoặc ảnh có độ phân giải thấp. Thư mục `data/cleaned` chứa dữ liệu đã qua bước làm sạch tự động và lọc trùng. Đây là thư mục chính được dùng để upload lên Kaggle. Thư mục `data/duplicates` lưu một phần ảnh trùng để phục vụ báo cáo. Thư mục `data/rejected` lưu ảnh bị loại trong quá trình review thủ công.

Việc chia thư mục theo từng label giúp notebook hoặc thư viện TensorFlow/Keras tự nhận diện nhãn dựa trên tên folder. Đây cũng là cấu trúc tương thích với nhiều công cụ xử lý ảnh phổ biến.

## 2.2. Nguồn dữ liệu và công cụ crawl

Nhóm sử dụng phương pháp tự crawl dữ liệu từ Internet. Các nguồn chính gồm:

- Naver Images thông qua AutoCrawler.
- DuckDuckGo Images thông qua package `ddgs`.
- Google Images được thử nghiệm nhưng không dùng làm nguồn chính do gặp CAPTCHA trong quá trình crawl.

AutoCrawler là repository hỗ trợ crawl ảnh từ các nguồn như Google và Naver. Tuy nhiên, khi chạy thực tế, Google có cơ chế chặn tự động bằng CAPTCHA, làm pipeline không ổn định. Vì vậy, nhóm điều chỉnh pipeline để ưu tiên Naver và DuckDuckGo. DuckDuckGo được dùng như nguồn fallback hoặc nguồn bổ sung khi số lượng ảnh của một label chưa đủ.

Thông tin cấu hình crawl được đặt trong `configs/labels.yaml`, gồm danh sách label, keyword cho từng label, nguồn crawl, số ảnh mục tiêu, ngưỡng kích thước ảnh và cấu hình lọc trùng. Cách tổ chức này giúp pipeline dễ tái chạy, dễ thay đổi keyword và dễ chứng minh quy trình thu thập dữ liệu trong báo cáo.

![FIG-CRAWL-02: Cấu hình label và keyword trong labels.yaml](PLACEHOLDER_FIG_CRAWL_02)

## 2.3. Chiến lược keyword

Mỗi label không chỉ dùng một keyword duy nhất. Nếu chỉ crawl bằng keyword như `car` hoặc `bus`, dữ liệu có thể bị trùng nhiều, góc chụp nghèo nàn và dễ chứa ảnh minh họa không thực tế. Vì vậy, nhóm mở rộng keyword theo nhiều hướng:

- Từ khóa tổng quát: `car`, `bus`, `truck`.
- Từ khóa theo bối cảnh: `car on road`, `bus in traffic`, `boat on water`.
- Từ khóa theo góc nhìn: `side view`, `front view`, `street photo`.
- Từ khóa theo biến thể: `electric car`, `school bus`, `cargo truck`, `rescue helicopter`.
- Từ khóa nhấn mạnh ảnh thật: `real car`, `vehicle photo`, `outdoor`.

Ví dụ với label `car`, các keyword gồm `car`, `sedan car`, `road car`, `vehicle car`, `family car`, `hatchback car`, `car on street`, `car side view`, `car in traffic`, `electric car`, `parked car`, `modern car`. Với label `bus`, các keyword gồm `city bus`, `public bus`, `coach bus`, `school bus`, `transit bus`, `double decker bus`, `airport bus`, `tour bus`, `bus terminal`.

Việc đa dạng keyword giúp dữ liệu có nhiều bối cảnh hơn và giảm tỷ lệ ảnh trùng. Đây là một kinh nghiệm quan trọng trong quá trình crawl ảnh thực tế.

## 2.4. Quy trình crawl dữ liệu

Quy trình crawl được xây dựng bằng script `crawl_images.py`. Script đọc file cấu hình YAML, lấy danh sách label và keyword, sau đó chạy crawl theo từng label. Ảnh sau khi tải về được gom vào đúng thư mục `data/raw/<label>/`. Mỗi ảnh được ghi metadata gồm label, keyword, source, đường dẫn file và thời điểm crawl.

Pipeline crawl có thêm cơ chế chỉ tập trung vào label còn thiếu dữ liệu. Khi một label đã gần đủ số lượng ảnh sạch, pipeline không tiếp tục crawl label đó nữa. Điều này giúp giảm mất cân bằng dữ liệu và tiết kiệm thời gian crawl. Trong quá trình chạy, nhóm nhận thấy có hiện tượng ảnh trùng tăng mạnh nếu lặp lại cùng keyword quá nhiều lần, vì vậy keyword được mở rộng và có chiến lược shuffle/rotate để tăng đa dạng.

![FIG-CRAWL-03: Log tiến độ crawl theo từng label](PLACEHOLDER_FIG_CRAWL_03)

## 2.5. Làm sạch dữ liệu

Ảnh crawl từ Internet có chất lượng không đồng nhất. Một số ảnh có thể bị lỗi khi tải về, ảnh không mở được, ảnh có kích thước quá nhỏ, ảnh grayscale, ảnh có alpha channel hoặc ảnh không phải đối tượng phương tiện. Do đó, bước làm sạch dữ liệu là bắt buộc.

Script `remove_corrupted_images.py` thực hiện các thao tác:

- Mở ảnh bằng Pillow để kiểm tra ảnh có đọc được hay không.
- Loại ảnh lỗi hoặc ảnh không decode được.
- Loại ảnh có kích thước nhỏ hơn `128x128`.
- Chuyển ảnh sang RGB.
- Lưu lại dưới định dạng JPEG.
- Đổi tên theo dạng `label_000001.jpg`, `label_000002.jpg`.
- Lưu ảnh sạch vào `data/cleaned/<label>/`.

Quy trình này giúp dữ liệu đầu vào của mô hình đồng nhất hơn. Việc đổi tên ảnh cũng giúp quản lý dữ liệu dễ hơn, tránh phụ thuộc vào tên file tải từ Internet vốn thường dài, chứa ký tự đặc biệt hoặc không có ý nghĩa.

![FIG-CLEAN-01: Ví dụ ảnh lỗi, ảnh quá nhỏ hoặc ảnh bị loại](PLACEHOLDER_FIG_CLEAN_01)

## 2.6. Lọc ảnh trùng bằng perceptual hashing

Một vấn đề lớn khi crawl ảnh là trùng lặp. Nhiều nguồn tìm kiếm trả về cùng một ảnh hoặc các phiên bản gần giống nhau với kích thước khác nhau. Nếu không lọc, mô hình có thể học thuộc ảnh thay vì học đặc trưng tổng quát. Ngoài ra, ảnh trùng giữa train và test có thể làm kết quả đánh giá bị ảo.

Nhóm dùng perceptual hashing, cụ thể là pHash hoặc dHash. Khác với hash truyền thống như MD5, perceptual hash cố gắng biểu diễn nội dung thị giác của ảnh. Hai ảnh gần giống nhau thường có hash gần nhau dù kích thước hoặc mức nén khác nhau.

Hamming distance được tính bằng số bit khác nhau giữa hai hash. Nếu khoảng cách Hamming nhỏ hơn hoặc bằng ngưỡng cấu hình, hai ảnh được xem là trùng hoặc gần trùng. Trong pipeline hiện tại, nhóm dùng:

- Hash method: `phash`.
- Hamming threshold: `4`.
- Chỉ so sánh trong cùng label: `within_label_only = true`.

Việc chỉ lọc trùng trong cùng label giúp tránh loại nhầm những ảnh có hình dạng giống nhau nhưng thuộc label khác nhau, ví dụ `car` và `minibus`. Đây là lựa chọn an toàn trong bối cảnh các lớp phương tiện có thể có hình dạng tương đồng.

![FIG-CLEAN-02: Ví dụ ảnh trùng hoặc gần trùng được phát hiện bằng pHash](PLACEHOLDER_FIG_CLEAN_02)

## 2.7. Lọc thủ công

Sau khi làm sạch tự động, dữ liệu vẫn cần lọc thủ công. Lý do là thuật toán không thể xác định chắc chắn ảnh có đúng nhãn hay không. Ví dụ, keyword `train` có thể trả về ga tàu hoặc đường ray không có tàu. Keyword `boat` có thể trả về ảnh phong cảnh biển có thuyền rất nhỏ. Keyword `bus` có thể trả về nội thất xe hoặc biển báo không phù hợp.

Nhóm sử dụng cơ chế review thủ công bằng grid ảnh. Ảnh được hiển thị theo từng label để thành viên kiểm tra nhanh. Các ảnh sai nhãn, ảnh có đối tượng quá nhỏ, ảnh bị che khuất hoặc ảnh không phù hợp sẽ được chuyển sang `data/rejected/<label>/`. Cả nhóm cùng tham gia bước này để giảm sai sót chủ quan.

![FIG-CLEAN-03: Giao diện/grid lọc thủ công dữ liệu](PLACEHOLDER_FIG_CLEAN_03)

## 2.8. Thống kê dữ liệu sau làm sạch

Sau quá trình crawl, làm sạch, lọc trùng và cân bằng dữ liệu, tập `data/cleaned` hiện có tổng cộng **10.767 ảnh** (đã bỏ nhãn `taxi`). Đây là số lượng vẫn thỏa yêu cầu hơn 10.000 mẫu. Bảng sau trình bày số lượng ảnh theo từng label sau khi cân bằng:

| Label | Số ảnh |
|---|---:|
| bicycle | 1200 |
| boat | 1200 |
| bus | 1200 |
| car | 1200 |
| helicopter | 1167 |
| minibus | 1200 |
| motorcycle | 1200 |
| train | 1200 |
| truck | 1200 |

![FIG-DATA-01: Biểu đồ phân bố số ảnh theo class](PLACEHOLDER_FIG_DATA_01)

Nhìn chung, dữ liệu sau cân bằng có phân bố khá đều (hầu hết các lớp ở mức 1200 ảnh). Lớp `helicopter` giữ mức 1167 ảnh do số lượng ảnh hợp lệ ban đầu thấp hơn các lớp khác, nhưng vẫn đủ lớn để huấn luyện. Notebook sử dụng stratified split để giữ phân bố class ổn định giữa train, validation và test.

## 2.9. Phân tích chi tiết về dữ liệu crawl

### 2.9.1. Vì sao dữ liệu crawl cần được kiểm soát chặt?

Dữ liệu crawl từ Internet khác với dữ liệu benchmark đã được chuẩn hóa. Trong một benchmark phổ biến, ảnh thường đã được kiểm duyệt, nhãn đã được xác nhận, định dạng ảnh tương đối ổn định và phân bố class có thể đã được thiết kế trước. Ngược lại, dữ liệu crawl bằng công cụ tìm kiếm ảnh thường phản ánh cách công cụ tìm kiếm hiểu keyword, không phản ánh hoàn toàn ý định của người xây dựng dataset. Ví dụ, keyword `bus` có thể trả về ảnh xe buýt thật, ảnh đồ chơi xe buýt, icon xe buýt, biển báo trạm xe buýt, ảnh bên trong xe buýt hoặc poster quảng cáo. Keyword `helicopter` có thể trả về ảnh trực thăng thật, mô hình đồ chơi, ảnh hoạt hình, biểu tượng vector hoặc ảnh có trực thăng rất nhỏ ở xa.

Vì vậy, quá trình crawl cần được xem là bước tạo dữ liệu thô, chưa phải dữ liệu huấn luyện cuối cùng. Nếu đưa trực tiếp toàn bộ ảnh raw vào train, mô hình có thể học các đặc trưng sai. Chẳng hạn, nếu nhiều ảnh `bus` có chữ quảng cáo lớn ở nền, mô hình có thể dựa vào chữ thay vì học hình dạng xe. Nếu nhiều ảnh `boat` là ảnh phong cảnh biển với thuyền nhỏ, mô hình có thể học nền nước thay vì học hình dạng thuyền. Những lỗi này làm mô hình có vẻ tốt trên test set nội bộ nhưng kém khi gặp ảnh thực tế.

Trong đồ án, nhóm xử lý vấn đề này bằng nhiều tầng kiểm soát. Tầng thứ nhất là mở rộng keyword để giảm thiên lệch do một keyword đơn lẻ. Tầng thứ hai là làm sạch tự động để loại ảnh lỗi và ảnh quá nhỏ. Tầng thứ ba là lọc trùng bằng perceptual hash để tránh lặp lại nhiều ảnh giống nhau. Tầng cuối cùng là lọc thủ công để kiểm tra nhãn và chất lượng hình ảnh.

### 2.9.2. Lý do giữ lại một phần ảnh duplicate cho báo cáo

Ban đầu, thư mục duplicate có số lượng rất lớn vì cùng một keyword được crawl nhiều lần và một số nguồn trả về ảnh giống nhau. Sau khi phát hiện vấn đề này, nhóm thay đổi chiến lược keyword và cấu hình dedup. Tuy nhiên, nhóm không xóa toàn bộ duplicate mà giữ lại một số mẫu đại diện. Việc giữ mẫu duplicate có hai lợi ích.

Thứ nhất, nó giúp minh họa rõ ràng trong báo cáo rằng dữ liệu crawl thật sự có vấn đề trùng lặp và nhóm đã xử lý vấn đề đó. Nếu chỉ mô tả bằng chữ, người đọc khó hình dung mức độ trùng ảnh. Khi có hình minh họa các ảnh gần giống nhau, phần giải thích về pHash, dHash và Hamming distance sẽ thuyết phục hơn.

Thứ hai, nó giúp nhóm có bằng chứng quy trình. Một báo cáo khoa học dữ liệu không chỉ trình bày kết quả cuối cùng mà còn cần mô tả các quyết định xử lý dữ liệu. Ảnh duplicate là một ví dụ tốt để chứng minh dữ liệu đã được kiểm tra và xử lý có hệ thống.

### 2.9.3. Tiêu chí lọc thủ công đề xuất

Khi lọc thủ công, nhóm nên thống nhất tiêu chí để tránh mỗi người lọc một kiểu. Một ảnh nên được giữ nếu phương tiện chính thuộc đúng label, vật thể đủ lớn, ảnh không quá mờ và không phải hình minh họa hoặc logo. Một ảnh nên bị loại nếu không có phương tiện, phương tiện quá nhỏ, có nhiều phương tiện thuộc nhiều class khác nhau mà không rõ đối tượng chính, ảnh hoạt hình, ảnh đồ chơi, ảnh render 3D không thực tế hoặc ảnh có watermark quá lớn che mất phương tiện.

Với các class dễ nhầm, cần thêm tiêu chí riêng. Với `minibus`, cần phân biệt với `bus` bằng kích thước và dạng thân xe; nếu xe quá lớn giống bus thành phố thì nên chuyển sang review. Với `truck`, nên giữ ảnh xe tải chở hàng, xe container, delivery truck; không nên giữ ảnh pickup nhỏ nếu dễ gây nhầm với car.

\newpage

# 3. TRÍCH XUẤT ĐẶC TRƯNG VÀ XỬ LÝ DỮ LIỆU

## 3.1. Đặc trưng ảnh được sử dụng

Trong bài toán phân loại ảnh bằng mạng CNN, đặc trưng không được thiết kế thủ công hoàn toàn như các bài toán dữ liệu bảng. Thay vào đó, ảnh đầu vào được đưa qua nhiều lớp convolution để mô hình tự học đặc trưng. Những lớp đầu thường học cạnh, góc, đường viền, vùng màu; các lớp giữa học texture và bộ phận vật thể; các lớp sâu hơn học đặc trưng ngữ nghĩa như bánh xe, thân xe, cửa kính, đầu tàu, cánh quạt trực thăng hoặc hình dáng tổng thể của phương tiện.

Tuy vậy, trước khi đưa ảnh vào mô hình, dữ liệu vẫn cần được chuẩn hóa để đảm bảo mô hình nhận được đầu vào đồng nhất. Các đặc trưng cơ bản được thống kê trong giai đoạn khám phá dữ liệu gồm:

- Đường dẫn ảnh.
- Nhãn ảnh.
- Width.
- Height.
- Channel.
- File size.
- Aspect ratio.
- Pixel values sau khi resize và normalize.

Những đặc trưng này không nhất thiết đều được đưa trực tiếp vào mô hình. Mô hình chính sử dụng tensor ảnh đã resize và normalize. Các đặc trưng như width, height, file size được dùng để phân tích dữ liệu, phát hiện bất thường và minh họa quá trình tiền xử lý.

![FIG-PRETRAIN-01: Ảnh mẫu theo từng class trước train](PLACEHOLDER_FIG_PRETRAIN_01)

## 3.2. Thống kê kích thước ảnh

Ảnh crawl có kích thước rất đa dạng. Một số ảnh chỉ lớn hơn ngưỡng tối thiểu 128 pixel, trong khi một số ảnh có độ phân giải vài nghìn pixel. Nếu đưa trực tiếp ảnh với kích thước khác nhau vào mô hình, batch training sẽ không thể hoạt động ổn định. Vì vậy, ảnh cần được resize về một kích thước cố định.

Trong notebook, nhóm chọn kích thước `224x224`. Đây là kích thước phổ biến cho nhiều mô hình CNN, bao gồm ResNet và MobileNet. Việc chọn 224x224 là sự cân bằng giữa khả năng giữ thông tin hình ảnh và chi phí tính toán. Ảnh lớn hơn có thể giữ nhiều chi tiết hơn nhưng làm train chậm hơn. Ảnh nhỏ hơn train nhanh hơn nhưng có thể mất chi tiết, đặc biệt với các lớp có hình dạng gần giống nhau.

![FIG-PRETRAIN-02: Histogram width/height của ảnh](PLACEHOLDER_FIG_PRETRAIN_02)

![FIG-PRETRAIN-03: Boxplot width/height/file size](PLACEHOLDER_FIG_PRETRAIN_03)

## 3.3. Chuẩn hóa pixel

Ảnh gốc có giá trị pixel trong khoảng 0 đến 255. Trước khi đưa vào mô hình, pixel được chuyển sang `float32` và đưa qua preprocessing phù hợp với Keras Applications. Chuẩn hóa này giúp quá trình tối ưu ổn định hơn. Nếu giữ giá trị pixel lớn, gradient có thể dao động mạnh hơn và mô hình khó học hơn.

Trong pipeline, validation set và test set chỉ được resize và normalize, không augmentation. Train set được áp dụng augmentation để tăng tính đa dạng dữ liệu. Việc tách rõ tiền xử lý giữa train và test giúp đánh giá công bằng hơn, vì test set đại diện cho dữ liệu chưa thấy.

![FIG-PRETRAIN-04: Ảnh trước và sau resize/normalize](PLACEHOLDER_FIG_PRETRAIN_04)

## 3.4. Data augmentation

Dữ liệu cần đủ đa dạng để tránh overfitting. Nhóm sử dụng augmentation cơ bản trên train set:

- Random horizontal flip.
- Random rotation.
- Random zoom.
- Random contrast.

Random flip phù hợp với đa số phương tiện vì xe nhìn từ trái sang phải hoặc phải sang trái vẫn là cùng một lớp. Random rotation nhỏ giúp mô hình chịu được ảnh nghiêng nhẹ. Random zoom giúp mô hình học tốt hơn khi vật thể ở gần hoặc xa. Random contrast giúp mô hình ít phụ thuộc vào điều kiện ánh sáng.

Tuy nhiên, augmentation cần vừa phải. Nếu rotation quá mạnh hoặc zoom quá lớn, ảnh có thể trở nên không thực tế. Với bài toán phương tiện giao thông, vật thể thường có hướng tương đối rõ, nên augmentation cực đoan có thể làm giảm chất lượng học.

![FIG-PRETRAIN-05: Minh họa data augmentation trên train set](PLACEHOLDER_FIG_PRETRAIN_05)

## 3.5. Trực quan hóa PCA/t-SNE trước train

Notebook có phần trực quan hóa đặc trưng bằng PCA hoặc t-SNE. Mục tiêu của bước này không phải để train mô hình chính, mà để hiểu sơ bộ mức độ phân tách của dữ liệu. Có thể dùng pixel feature sau khi resize nhỏ để giảm chiều và vẽ lên mặt phẳng 2D.

Nếu các điểm của từng class tạo thành cụm tương đối riêng, bài toán có khả năng phân loại tốt hơn. Nếu các class trộn lẫn mạnh, mô hình có thể gặp khó khăn, đặc biệt với các lớp tương đồng. Với dữ liệu phương tiện, một số class như `boat` và `helicopter` thường dễ tách hơn vì hình dạng và bối cảnh khác biệt. Ngược lại, `car`, `minibus`, `bus` và `truck` có thể bị trộn lẫn do cùng xuất hiện trên đường và có hình dạng hộp hoặc thân xe tương tự.

![FIG-PRETRAIN-06: PCA hoặc t-SNE đặc trưng ảnh trước train](PLACEHOLDER_FIG_PRETRAIN_06)

## 3.6. Chia tập train/validation/test

Notebook sử dụng stratified split theo label với tỉ lệ 70/15/15. Stratified split đảm bảo mỗi tập có phân bố class gần giống nhau. Điều này quan trọng vì nếu một class xuất hiện quá ít trong test set, metric đánh giá có thể thiếu ổn định.

Ba tập dữ liệu có vai trò khác nhau:

- Train set: dùng để cập nhật trọng số mô hình.
- Validation set: dùng để theo dõi quá trình train, chọn best model và early stopping.
- Test set: dùng để đánh giá cuối cùng sau khi mô hình đã được chọn.

Trong quá trình huấn luyện chính, nhóm sử dụng `data/cleaned` làm nguồn dữ liệu đầu vào và để notebook tạo split trực tiếp bằng stratified split. Cách làm này đảm bảo train, validation và test được sinh ra từ phiên bản dữ liệu cleaned mới nhất.

![FIG-SPLIT-01: Biểu đồ so sánh phân bố class giữa train/validation/test](PLACEHOLDER_FIG_SPLIT_01)

## 3.7. Kiểm tra distribution shift

Distribution shift là hiện tượng phân bố dữ liệu giữa train và test khác nhau đáng kể. Ví dụ, train set có nhiều ảnh xe buýt ban ngày nhưng test set có nhiều ảnh xe buýt ban đêm; hoặc train set có nhiều ảnh `truck` ở cự ly gần nhưng test set có nhiều ảnh `truck` ở xa. Trong đồ án này, nhóm kiểm tra distribution shift ở mức cơ bản bằng cách so sánh phân bố class giữa các tập và so sánh một số thống kê ảnh như width, height, file size.

Nếu stratified split hoạt động đúng, tỷ lệ class giữa train, validation và test sẽ gần nhau. Tuy nhiên, vì ảnh được crawl từ Internet, vẫn có thể có shift về bối cảnh, góc chụp, ánh sáng hoặc nguồn ảnh. Đây là hạn chế tự nhiên của dữ liệu crawl và cần được thảo luận trong phần kết quả.

\newpage

# 4. MÔ HÌNH HÓA DỮ LIỆU

## 4.1. Lý do chọn mạng CNN

Convolutional Neural Network là kiến trúc phù hợp với dữ liệu ảnh vì nó khai thác cấu trúc không gian của ảnh. Thay vì xem ảnh như một vector phẳng, CNN dùng kernel trượt qua ảnh để học các pattern cục bộ. Cơ chế convolution giúp mô hình nhận diện cạnh, texture, hình dạng và các bộ phận của vật thể ở nhiều vị trí khác nhau.

Trong bài toán phân loại phương tiện, các đặc trưng quan trọng có thể nằm ở nhiều vùng ảnh: bánh xe, thân xe, kính chắn gió, cửa xe, đầu tàu, cánh quạt trực thăng, buồng lái, đường ray, mặt nước hoặc bối cảnh đường phố. CNN có khả năng kết hợp các đặc trưng cục bộ thành đặc trưng toàn cục để đưa ra nhãn cuối.

## 4.2. Cấu hình huấn luyện 2 giai đoạn

Notebook huấn luyện MobileNetV2 và ResNet50 theo 2 giai đoạn. Ở giai đoạn đầu, backbone được freeze và chỉ classifier head mới được cập nhật. Giai đoạn này giúp lớp phân loại cuối học nhanh quan hệ giữa đặc trưng ảnh và 9 nhãn phương tiện. Ở giai đoạn sau, một phần các lớp cuối của backbone được mở lại và huấn luyện với learning rate nhỏ để tinh chỉnh đặc trưng cho dữ liệu của nhóm.

Điều này có các hệ quả:

- Giai đoạn đầu thường hội tụ nhanh hơn vì classifier head có ít tham số.
- Giai đoạn sau cần learning rate nhỏ để tránh làm dao động các đặc trưng đã học.
- Batch Normalization trong backbone được giữ ổn định khi mở lớp để giảm rủi ro validation accuracy dao động mạnh.
- Augmentation nhẹ và MixUp giúp giảm overfitting trên dữ liệu crawl.
- Kết quả cuối vẫn phụ thuộc rất mạnh vào chất lượng dữ liệu cleaned.

Trong notebook, mô hình dùng optimizer Adam, focal loss dạng categorical, learning rate `1e-3` ở giai đoạn warm-up và `1e-5` ở giai đoạn mở lớp. Có callback `TerminateOnNaN`, `EarlyStopping`, `ReduceLROnPlateau` và `ModelCheckpoint` để lưu best model theo validation accuracy. Notebook cũng hỗ trợ MixUp nhẹ (`alpha=0.15`) và class weight có cap, nhưng mặc định không bật class weight để tránh làm loss mất ổn định.

Khi chạy trên Kaggle với GPU T4x2, notebook sử dụng `tf.distribute.MirroredStrategy` để phân phối quá trình train trên hai GPU. Notebook giữ chính sách tính toán `float32` để ưu tiên độ ổn định vì pipeline có augmentation, MixUp và focal loss.

## 4.3. MobileNetV2

MobileNet là họ mô hình CNN được thiết kế để giảm số lượng tham số và phép tính. Ý tưởng quan trọng nhất là depthwise separable convolution. Trong convolution thường, một kernel học đồng thời cả chiều không gian và chiều kênh. Trong depthwise separable convolution, quá trình này được tách thành hai bước:

1. Depthwise convolution: mỗi kênh đầu vào được lọc riêng bằng một kernel không gian.
2. Pointwise convolution: dùng convolution 1x1 để trộn thông tin giữa các kênh.

Cách tách này giúp giảm đáng kể số phép nhân và số tham số so với convolution thường. Với bài toán phân loại phương tiện, MobileNetV2 có ưu điểm là train nhanh hơn, nhẹ hơn và phù hợp nếu muốn triển khai trên thiết bị hạn chế tài nguyên. Tuy nhiên, vì số tham số ít hơn, khả năng biểu diễn của MobileNetV2 có thể thấp hơn ResNet50 trong một số trường hợp.

Kiến trúc MobileNetV2 trong notebook dùng `tf.keras.applications.MobileNetV2`, gồm:

- Input shape: `224x224x3`.
- `include_top=False` để bỏ classifier mặc định.
- Inverted residual block.
- Depthwise convolution và pointwise convolution.
- Batch normalization và ReLU/ReLU6 theo kiến trúc chuẩn.
- Global average pooling.
- Dropout.
- Dense output với softmax.

Chi tiết kiến trúc MobileNetV2 được triển khai như sau:

| Thành phần | Cấu hình | Output/Filter chính | Activation | Ghi chú |
|---|---|---:|---|---|
| Input | Ảnh RGB đã resize | 224x224x3 | - | Pixel được preprocess theo kiểu `tf`, khoảng [-1,1] |
| Base architecture | `MobileNetV2(include_top=False)` | Feature map cuối | ReLU/ReLU6 | Kiến trúc chuẩn Keras Applications |
| Main block | Inverted residual + depthwise separable convolution | Theo MobileNetV2 chuẩn | ReLU/ReLU6 | Giảm chi phí tính toán so với CNN thường |
| Pooling | GlobalAveragePooling2D | Vector đặc trưng | - | Giảm số tham số so với Flatten |
| Regularization | Dropout | 0.35 | - | Giảm overfitting |
| Classifier | Dense | 9 nodes | Softmax | Dự đoán xác suất cho 9 class |

Mô hình được train theo 2 giai đoạn: trước hết huấn luyện classifier head, sau đó mở một phần các lớp cuối của backbone với learning rate nhỏ để tối ưu cho dữ liệu phương tiện.

![FIG-MODEL-01: Kiến trúc tổng quan MobileNetV2](PLACEHOLDER_FIG_MODEL_01)

## 4.4. ResNet50

ResNet được thiết kế để giải quyết vấn đề khó train mạng sâu. Khi số lớp tăng, gradient có thể suy giảm trong quá trình lan truyền ngược, khiến các lớp đầu học chậm. ResNet sử dụng residual block với skip connection. Thay vì học trực tiếp hàm ánh xạ `H(x)`, block học phần dư `F(x)` và cộng lại với đầu vào `x`.

Biểu diễn đơn giản:

```text
Output = F(x) + x
```

Skip connection giúp gradient có đường đi ngắn hơn khi lan truyền ngược, làm quá trình train ổn định hơn. Với dữ liệu ảnh phương tiện, ResNet50 có thể học đặc trưng sâu hơn và phức tạp hơn MobileNetV2. Tuy nhiên, ResNet50 có nhiều tham số hơn, train chậm hơn và dễ overfit nếu dữ liệu chưa đủ đa dạng.

Kiến trúc ResNet50 trong notebook dùng `tf.keras.applications.ResNet50`, gồm:

- Input shape: `224x224x3`.
- `include_top=False` để bỏ classifier mặc định.
- Stem convolution 7x7 và max pooling theo kiến trúc chuẩn.
- Các stage bottleneck residual block với skip connection.
- Batch normalization và ReLU theo kiến trúc chuẩn.
- Global average pooling.
- Dropout.
- Dense output với softmax.

Chi tiết kiến trúc ResNet50 được triển khai như sau:

| Thành phần | Cấu hình | Filter/Block | Activation | Ghi chú |
|---|---|---:|---|---|
| Input | Ảnh RGB đã resize | 224x224x3 | - | Pixel được preprocess theo kiểu `tf`, khoảng [-1,1] |
| Base architecture | `ResNet50(include_top=False)` | Feature map cuối | ReLU | Kiến trúc chuẩn Keras Applications |
| Main block | Bottleneck residual block | Theo ResNet50 chuẩn | ReLU | Dùng skip connection để cải thiện lan truyền gradient |
| Pooling | GlobalAveragePooling2D | Vector đặc trưng | - | Gom đặc trưng không gian |
| Regularization | Dropout | 0.35 | - | Giảm overfitting |
| Classifier | Dense | 9 nodes | Softmax | Dự đoán xác suất cho 9 class |

Mô hình ResNet50 được train theo 2 giai đoạn tương tự MobileNetV2. Vì ResNet50 có nhiều tham số hơn, phần mở lớp được thực hiện với learning rate nhỏ để giữ ổn định quá trình huấn luyện.

![FIG-MODEL-02: Kiến trúc tổng quan ResNet50](PLACEHOLDER_FIG_MODEL_02)

## 4.5. Tham số huấn luyện

Các tham số chính trong notebook:

| Nhóm | Tham số | Giá trị | Ý nghĩa |
|---|---|---|---|
| Dữ liệu | Image size | 224x224 | Kích thước đầu vào thống nhất cho hai mô hình |
| Dữ liệu | Channel | 3 | Ảnh RGB |
| Dữ liệu | Normalize | Pixel / 255.0 | Đưa giá trị pixel về [0,1] |
| Dữ liệu | Split | 70% train, 15% validation, 15% test | Chia stratified theo label |
| Dữ liệu | Per-replica batch size | 32 | Số ảnh trên mỗi GPU |
| Dữ liệu | Global batch size | 32 x số GPU, T4x2 là 64 | Số ảnh của một batch train sau khi phân phối |
| Huấn luyện | Epochs | 35 | 5 epoch warm-up + 30 epoch mở một phần backbone |
| Huấn luyện | Optimizer | Adam | Thuật toán cập nhật trọng số |
| Huấn luyện | Learning rate | 1e-3 và 1e-5 | 1e-3 cho warm-up, 1e-5 cho giai đoạn mở lớp |
| Huấn luyện | Loss function | Categorical focal loss | Phù hợp với one-hot label và dữ liệu có class dễ nhầm |
| Huấn luyện | Metrics | Accuracy | Theo dõi độ chính xác trong train/validation |
| Huấn luyện | MixUp | Alpha 0.15 | Tăng khả năng tổng quát, giảm overfitting |
| Huấn luyện | Sample weight/Class weight | Có hỗ trợ cap, mặc định tắt | Tránh weight quá lớn gây mất ổn định loss |
| Tăng tốc GPU | Distribution strategy | MirroredStrategy khi có nhiều GPU | Train song song trên T4x2 |
| Tăng tốc GPU | Precision policy | `float32` | Ưu tiên ổn định số học với augmentation, MixUp và focal loss |
| Augmentation | RandomFlip | Horizontal | Lật ngang ảnh train |
| Augmentation | RandomRotation | 0.04 | Xoay nhẹ ảnh train |
| Augmentation | RandomZoom | 0.08 | Phóng to/thu nhỏ nhẹ ảnh train |
| Augmentation | RandomContrast | 0.08 | Thay đổi tương phản nhẹ |
| Callback | ModelCheckpoint | Monitor `val_accuracy`, save best only | Lưu model tốt nhất trên validation set |
| Callback | TerminateOnNaN | Dừng khi loss NaN | Tránh chạy phí GPU nếu cấu hình train lỗi |
| Callback | EarlyStopping | Monitor `val_accuracy`, patience 10 | Dừng sớm khi validation accuracy không cải thiện |
| Callback | ReduceLROnPlateau | Monitor `val_accuracy`, factor 0.3, patience 4, min_lr 1e-6 | Giảm learning rate khi validation accuracy chững lại |

Hai mô hình có một số thông số riêng:

| Thông số riêng | MobileNetV2 | ResNet50 |
|---|---|---|
| Kiến trúc chuẩn | `tf.keras.applications.MobileNetV2` | `tf.keras.applications.ResNet50` |
| Classifier gốc | Bỏ bằng `include_top=False` | Bỏ bằng `include_top=False` |
| Block chính | Inverted residual + depthwise separable convolution | Bottleneck residual block |
| Batch Normalization | Có, theo kiến trúc MobileNetV2 chuẩn | Có, theo kiến trúc ResNet50 chuẩn |
| Activation | ReLU/ReLU6 | ReLU |
| Pooling cuối | GlobalAveragePooling2D | GlobalAveragePooling2D |
| Dropout | 0.35 | 0.35 |
| Output | Dense 9, softmax | Dense 9, softmax |
| Mục tiêu thiết kế | Nhẹ, ít tham số, train nhanh hơn | Học đặc trưng sâu hơn nhờ skip connection |

Nhóm sử dụng `ModelCheckpoint` để lưu model tốt nhất theo validation accuracy. Điều này cần thiết vì model ở epoch cuối chưa chắc là model tốt nhất. Nếu validation accuracy đạt đỉnh ở epoch giữa rồi giảm do overfitting, checkpoint giúp giữ lại model tốt nhất.

## 4.6. Kết quả train MobileNet

Kết quả huấn luyện MobileNet được tổng hợp bằng biểu đồ loss/accuracy và các chỉ số đánh giá trên test set.

![FIG-TRAIN-01: Training loss/accuracy MobileNet](PLACEHOLDER_FIG_TRAIN_01)

| Metric | Giá trị |
|---|---:|
| Best validation accuracy | PLACEHOLDER_MOBILENET_BEST_VAL_ACC |
| Test accuracy | PLACEHOLDER_MOBILENET_TEST_ACC |
| Precision | PLACEHOLDER_MOBILENET_PRECISION |
| Recall | PLACEHOLDER_MOBILENET_RECALL |
| F1-score | PLACEHOLDER_MOBILENET_F1 |
| Number of parameters | PLACEHOLDER_MOBILENET_PARAMS |
| Training time | PLACEHOLDER_MOBILENET_TRAIN_TIME |

Biểu đồ loss/accuracy cho thấy mức độ hội tụ của MobileNet trong quá trình huấn luyện. Khi train accuracy tăng nhưng validation accuracy không tăng tương ứng, mô hình có dấu hiệu overfitting. Khi cả train và validation accuracy đều thấp, mô hình có xu hướng underfitting hoặc dữ liệu còn khó. Khi validation accuracy tăng đều và loss giảm ổn định, quá trình train diễn ra tốt.

## 4.7. Kết quả train ResNet

![FIG-TRAIN-02: Training loss/accuracy ResNet](PLACEHOLDER_FIG_TRAIN_02)

| Metric | Giá trị |
|---|---:|
| Best validation accuracy | PLACEHOLDER_RESNET_BEST_VAL_ACC |
| Test accuracy | PLACEHOLDER_RESNET_TEST_ACC |
| Precision | PLACEHOLDER_RESNET_PRECISION |
| Recall | PLACEHOLDER_RESNET_RECALL |
| F1-score | PLACEHOLDER_RESNET_F1 |
| Number of parameters | PLACEHOLDER_RESNET_PARAMS |
| Training time | PLACEHOLDER_RESNET_TRAIN_TIME |

ResNet50 có khả năng học đặc trưng sâu hơn, nhưng cũng cần thời gian train dài hơn MobileNetV2. Nếu dữ liệu chưa đủ đa dạng, ResNet50 vẫn có nguy cơ học thuộc dữ liệu train, vì vậy checkpoint theo validation accuracy và augmentation nhẹ rất quan trọng.

## 4.8. Confusion matrix và classification report

Confusion matrix giúp quan sát mô hình nhầm class nào với class nào. Đây là công cụ quan trọng hơn accuracy tổng quát vì accuracy chỉ cho biết tỷ lệ đúng chung, không chỉ ra lỗi cụ thể. Với bài toán này, các lỗi nhầm lẫn có ý nghĩa thực tế:

- `minibus` bị nhầm thành `bus` vì cùng là xe chở khách.
- `truck` bị nhầm thành `bus` nếu góc chụp chỉ thấy thân xe lớn.
- `motorcycle` bị nhầm với `bicycle` nếu ảnh mờ hoặc phương tiện ở xa.

![FIG-EVAL-01: Confusion matrix MobileNet](PLACEHOLDER_FIG_EVAL_01)

![FIG-EVAL-02: Confusion matrix ResNet](PLACEHOLDER_FIG_EVAL_02)

## 4.9. So sánh MobileNet và ResNet

So sánh MobileNetV2 và ResNet50 là phần trọng tâm của báo cáo. MobileNetV2 đại diện cho hướng mô hình nhẹ, ít tham số, train nhanh hơn. ResNet50 đại diện cho hướng mô hình sâu hơn, có skip connection và khả năng biểu diễn mạnh hơn. Kết quả so sánh phản ánh tương quan giữa kiến trúc, dữ liệu và quá trình tối ưu.

| Model | Test Accuracy | Precision | Recall | F1-score | Number of parameters | Training time |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| ResNet50 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

![FIG-EVAL-03: Bar chart so sánh metric MobileNet và ResNet](PLACEHOLDER_FIG_EVAL_03)

Bảng so sánh cho thấy sự khác nhau giữa hai hướng kiến trúc. Nếu ResNet50 có F1-score cao hơn nhưng training time lớn hơn nhiều, kết quả thể hiện trade-off giữa hiệu quả và chi phí. Nếu MobileNetV2 đạt kết quả gần tương đương ResNet50, MobileNetV2 là lựa chọn hợp lý hơn cho triển khai thực tế. Nếu độ chính xác chưa đạt ngưỡng 85%, nguyên nhân được xem xét từ chất lượng dữ liệu, số epoch, augmentation và cấu hình learning rate.

## 4.10. Trực quan hóa dự đoán đúng và sai

Notebook đã bổ sung phần lấy mỗi label 5 ảnh dự đoán đúng trên test set và hiển thị ảnh dự đoán sai. Đây là phần rất hữu ích để đánh giá định tính. Metric số học cho biết mô hình đúng bao nhiêu, còn ảnh dự đoán đúng/sai cho biết mô hình sai vì lý do gì.

![FIG-PRED-01: 5 ảnh dự đoán đúng mỗi label của MobileNet](PLACEHOLDER_FIG_PRED_01)

![FIG-PRED-02: Ảnh dự đoán sai của MobileNet](PLACEHOLDER_FIG_PRED_02)

![FIG-PRED-03: 5 ảnh dự đoán đúng mỗi label của ResNet](PLACEHOLDER_FIG_PRED_03)

![FIG-PRED-04: Ảnh dự đoán sai của ResNet](PLACEHOLDER_FIG_PRED_04)

Các ảnh dự đoán sai được phân tích theo những nguyên nhân chính sau:

- Ảnh có nhiều phương tiện trong cùng khung hình.
- Phương tiện chính quá nhỏ hoặc bị che khuất.
- Góc chụp khiến hình dạng class bị biến đổi.
- Ảnh có watermark, chữ, logo hoặc nền gây nhiễu.
- Nhãn gốc có thể chưa chính xác.
- Hai class có bản chất gần nhau.

## 4.11. Artifact sau train

Notebook lưu các artifact quan trọng trong thư mục `/kaggle/working/training_artifacts`, gồm:

- `mobilenet_best.keras`.
- `mobilenet_final.keras`.
- `resnet_best.keras`.
- `resnet_final.keras`.
- `mobilenet_history.csv`.
- `resnet_history.csv`.
- `model_comparison.csv`.
- `*_classification_report.csv`.
- `*_test_predictions.csv`.
- `*_test_probabilities.npy`.
- Confusion matrix và training curve dạng ảnh PNG.

Các artifact này giúp nhóm tái kiểm tra kết quả, viết báo cáo, trình bày mô hình tốt nhất và xem lại dự đoán trên từng ảnh test.


## 4.12. Phân tích chi tiết về MobileNet

### 4.12.1. Convolution thường và chi phí tính toán

Trong một convolution 2D thông thường, mỗi filter có kích thước không gian và chiều kênh. Nếu input có `M` kênh, output có `N` kênh, kernel kích thước `K x K`, thì số tham số của lớp convolution là `K x K x M x N`. Khi ảnh lớn và số kênh tăng, số phép tính tăng rất nhanh. Điều này làm mô hình nặng, train chậm và khó triển khai trên thiết bị hạn chế tài nguyên.

MobileNet giảm chi phí bằng cách tách convolution thành depthwise convolution và pointwise convolution. Depthwise convolution dùng một filter riêng cho mỗi kênh input, nên số tham số là `K x K x M`. Pointwise convolution dùng kernel `1 x 1` để trộn kênh, số tham số là `M x N`. Tổng số tham số là `K x K x M + M x N`, thường nhỏ hơn nhiều so với `K x K x M x N`.

Ý tưởng này đặc biệt phù hợp khi cần mô hình nhẹ. Trong bối cảnh đề tài, MobileNetV2 được dùng để kiểm tra xem một kiến trúc ít tham số có đủ khả năng học phân loại 9 loại phương tiện hay không. Nếu MobileNetV2 đạt kết quả gần ResNet50, điều đó cho thấy dữ liệu có đặc trưng khá rõ và mô hình nhẹ vẫn có thể hoạt động tốt. Nếu MobileNetV2 kém hơn nhiều, có thể dữ liệu cần mô hình có khả năng biểu diễn mạnh hơn.

### 4.12.2. Vai trò của Batch Normalization và ReLU

Batch Normalization được dùng sau các lớp convolution để ổn định phân bố activation trong quá trình train. Khi mở một phần backbone ở giai đoạn sau, các lớp Batch Normalization trong backbone được giữ ổn định để tránh validation accuracy dao động mạnh. Điều này giúp quá trình huấn luyện trên Kaggle ổn định hơn.

ReLU activation đưa tính phi tuyến vào mô hình. Nếu chỉ gồm các lớp tuyến tính, dù xếp nhiều lớp, mô hình vẫn tương đương một phép biến đổi tuyến tính lớn. ReLU giúp mạng học được các ranh giới phức tạp hơn. Với ảnh phương tiện, ranh giới giữa class không tuyến tính đơn giản, vì vậy activation phi tuyến là bắt buộc.

### 4.12.3. Global Average Pooling

Thay vì flatten toàn bộ feature map rồi đưa vào dense layer lớn, mô hình dùng Global Average Pooling. Lớp này lấy trung bình mỗi kênh feature map, chuyển tensor không gian thành vector gọn hơn. Cách này giảm số tham số, giảm overfitting và phù hợp với kiến trúc CNN hiện đại. Trong MobileNet, Global Average Pooling giúp giữ mô hình nhẹ đúng với triết lý thiết kế của nó.

### 4.12.4. Nhận xét đánh giá MobileNet

Kết quả MobileNet được đánh giá theo các tiêu chí sau:

- Thời gian train của MobileNet so với ResNet.
- Mức giảm số tham số của MobileNet.
- Chênh lệch Accuracy và F1-score so với ResNet.
- Các class MobileNet nhầm nhiều nhất.
- Dấu hiệu underfitting hoặc overfitting của MobileNet.
- Độ ổn định của validation accuracy.

MobileNet có lợi thế khi thời gian train ngắn và số tham số thấp. Trong trường hợp kết quả chỉ thấp hơn ResNet một phần nhỏ, MobileNet phù hợp hơn cho các bối cảnh cần mô hình gọn nhẹ. Nếu kết quả thấp rõ rệt, nguyên nhân thường nằm ở dữ liệu phức tạp, nhiễu nhãn hoặc năng lực biểu diễn của kiến trúc chưa đủ.

## 4.13. Phân tích chi tiết về ResNet

### 4.13.1. Vấn đề khi mạng quá sâu

Trong lý thuyết, tăng số lớp có thể giúp mô hình học đặc trưng phức tạp hơn. Tuy nhiên, trong thực tế, mạng sâu khó train hơn. Gradient khi lan truyền ngược qua nhiều lớp có thể bị suy giảm hoặc biến đổi, khiến các lớp đầu học rất chậm. Ngoài ra, nếu thêm nhiều lớp nhưng quá trình tối ưu không tốt, mô hình sâu hơn có thể cho kết quả kém hơn mô hình nông hơn.

ResNet giải quyết vấn đề này bằng skip connection. Thay vì bắt mỗi block học toàn bộ ánh xạ từ input sang output, block chỉ cần học phần hiệu chỉnh so với input. Nếu một block không cần biến đổi nhiều, nó có thể học gần với hàm zero và để input đi qua skip connection. Điều này giúp việc tối ưu mạng sâu dễ hơn.

### 4.13.2. Residual block trong bài toán phương tiện

Trong ảnh phương tiện, đặc trưng có nhiều mức. Ở mức thấp, mô hình học cạnh, màu, texture. Ở mức trung bình, mô hình có thể học bánh xe, cửa xe, cánh quạt, thân tàu, đường ray. Ở mức cao, mô hình học hình dạng tổng thể của phương tiện. Residual block giúp mạng sâu kết hợp nhiều mức đặc trưng mà không làm gradient bị nghẽn quá mạnh.

Đối với các class dễ nhầm như `bus`, `minibus` và `truck`, đặc trưng phân biệt có thể nằm ở tỷ lệ thân xe, số cửa, kích thước cửa sổ, hình dạng khoang chở hàng hoặc bối cảnh. Những đặc trưng này có thể cần lớp sâu hơn để học tốt. Vì vậy, ResNet được kỳ vọng có lợi thế so với MobileNet nếu dữ liệu đủ tốt.

### 4.13.3. Rủi ro overfitting

ResNet có khả năng biểu diễn mạnh hơn nhưng cũng có nguy cơ overfitting. Nếu train accuracy tăng rất cao trong khi validation accuracy thấp hoặc dao động, mô hình có thể đang học thuộc train set. Điều này đặc biệt dễ xảy ra khi dữ liệu có ảnh trùng, ảnh gần trùng hoặc số lượng ảnh mỗi class chưa đủ đa dạng. Bởi vậy, bước dedup và augmentation có vai trò quan trọng.

Trong notebook, `EarlyStopping` và `ModelCheckpoint` giúp giảm rủi ro dùng model ở epoch cuối khi model đã overfit. `ReduceLROnPlateau` giúp giảm learning rate nếu validation accuracy không cải thiện, từ đó hỗ trợ mô hình hội tụ tốt hơn.

### 4.13.4. Nhận xét đánh giá ResNet

Kết quả ResNet được đánh giá theo các tiêu chí sau:

- Mức cải thiện Accuracy và F1-score so với MobileNet.
- Chi phí thời gian train của ResNet.
- Khả năng giảm nhầm lẫn ở các class khó trên confusion matrix.
- Dấu hiệu overfitting của ResNet.
- Mức đánh đổi giữa hiệu quả phân loại và chi phí tính toán.

Các tiêu chí này giúp phần so sánh không chỉ dừng ở bảng số liệu mà còn thể hiện rõ ý nghĩa thực nghiệm của từng mô hình.

\newpage

# 5. KẾT LUẬN

## 5.1. Kết quả đạt được

Đồ án đã xây dựng được pipeline tương đối hoàn chỉnh cho bài toán phân loại phương tiện giao thông từ ảnh. Nhóm đã tự thu thập dữ liệu bằng crawl ảnh từ Internet, tổ chức dữ liệu theo label, làm sạch dữ liệu, lọc trùng bằng perceptual hashing, thống kê dữ liệu và chuẩn bị notebook Kaggle để train MobileNet và ResNet từ đầu.

Tập dữ liệu cleaned hiện có **10.767 ảnh**, vượt yêu cầu hơn 10.000 mẫu. Dữ liệu gồm 9 class và được tổ chức theo cấu trúc phù hợp với image classification. Notebook chính có đầy đủ các bước: dataset overview, cleaning summary, train/validation/test split, preprocessing, feature visualization, MobileNetV2, ResNet50, evaluation, lưu best model và trực quan hóa dự đoán đúng/sai.

## 5.2. Hạn chế

Hạn chế lớn nhất của đề tài là dữ liệu crawl từ Internet không đồng đều. Một số ảnh có thể sai nhãn, có nhiều vật thể, có watermark hoặc không tập trung vào phương tiện cần phân loại. Dù đã có lọc tự động và lọc thủ công, dữ liệu vẫn có thể còn nhiễu.

Hạn chế thứ hai là bộ dữ liệu crawl vẫn có độ nhiễu nhất định. Với 10.767 ảnh tự thu thập, kết quả có thể bị ảnh hưởng bởi ảnh sai nhãn, ảnh nhiều vật thể hoặc ảnh có bối cảnh quá khác nhau giữa train và test. Đây là điểm cần được thảo luận rõ trong báo cáo cuối.

Hạn chế thứ ba là vẫn có chênh lệch nhẹ giữa các lớp sau cân bằng: đa số lớp ở mức 1200 ảnh, trong khi `helicopter` ở mức 1167 ảnh. Mức lệch này nhỏ nhưng vẫn có thể ảnh hưởng nhẹ đến precision/recall của từng class.

## 5.3. Hướng cải thiện

Các hướng cải thiện gồm:

- Crawl bổ sung cho các class còn ít như `helicopter`, đồng thời tiếp tục lọc tay để giữ chất lượng nhãn.
- Lọc thủ công kỹ hơn, đặc biệt các ảnh sai nhãn hoặc ảnh có nhiều phương tiện.
- Tăng số epoch và điều chỉnh learning rate schedule.
- Thử optimizer khác như SGD with momentum.
- Tăng augmentation có kiểm soát.
- Cân bằng class bằng oversampling hoặc giới hạn số mẫu mỗi class.
- Thử kiến trúc nhẹ hơn hoặc sâu hơn tùy kết quả ban đầu.
- Thực hiện thêm thí nghiệm với các cấu hình learning rate khác nhau để kiểm tra độ ổn định của kết quả.

## 5.4. Nhận xét cuối

Qua đồ án, nhóm nhận thấy chất lượng dữ liệu có ảnh hưởng rất lớn đến kết quả mô hình. Với dữ liệu crawl, phần khó không chỉ nằm ở train model mà còn nằm ở việc thu thập, lọc, chuẩn hóa và kiểm soát chất lượng dữ liệu. MobileNet và ResNet là hai kiến trúc có triết lý khác nhau: MobileNet tối ưu tính gọn nhẹ, ResNet tối ưu khả năng học sâu. Việc so sánh hai mô hình giúp nhóm hiểu rõ hơn mối quan hệ giữa kiến trúc, dữ liệu và hiệu quả phân loại.

\newpage

# 6. TÀI LIỆU THAM KHẢO

[1] K. He, X. Zhang, S. Ren, J. Sun, “Deep Residual Learning for Image Recognition,” IEEE Conference on Computer Vision and Pattern Recognition, 2016.  

[2] A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” arXiv preprint, 2017.  

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” IEEE Conference on Computer Vision and Pattern Recognition, 2018.  

[4] TensorFlow/Keras Documentation, “Keras API reference,” https://keras.io/  

[5] TensorFlow Documentation, “Image classification,” https://www.tensorflow.org/  

[6] Yoongi Kim, “AutoCrawler,” GitHub repository, https://github.com/YoongiKim/AutoCrawler  

[7] Python ImageHash documentation and package information, perceptual image hashing implementation.  

[8] DuckDuckGo search package `ddgs`, Python package for DuckDuckGo search and image retrieval.  

[9] Scikit-learn documentation, metrics, train_test_split, PCA, t-SNE utilities, https://scikit-learn.org/  

\newpage

# PHỤ LỤC

## Phụ lục A. Danh mục hình minh họa trong báo cáo

Các hình minh họa được đánh mã theo nhóm nội dung để dễ đối chiếu giữa báo cáo, notebook và thư mục kết quả thực nghiệm. Bảng sau tổng hợp các nhóm hình chính được sử dụng trong báo cáo.

| Nhóm hình | Mã hình | Nội dung minh họa | Nguồn đối chiếu |
|---|---|---|---|
| Crawl dữ liệu | FIG-CRAWL | Màn hình crawl, cấu hình keyword, log tiến độ | Terminal, `labels.yaml`, `pipeline_progress.csv` |
| Làm sạch dữ liệu | FIG-CLEAN | Ảnh lỗi, ảnh nhỏ, ảnh trùng, giao diện review | `data/rejected`, `data/duplicates`, `manual_review.html` |
| Thống kê dữ liệu | FIG-DATA | Phân bố class, thống kê số lượng mẫu | `dataset_summary.csv`, `reports/figures` |
| Trực quan hóa trước train | FIG-PRETRAIN | Ảnh mẫu, histogram, boxplot, resize, normalize, augmentation, PCA/t-SNE | Notebook Section 4, 7, 8 |
| Chia dữ liệu | FIG-SPLIT | Phân bố train/validation/test | Notebook Section 6 |
| Kiến trúc mô hình | FIG-MODEL | MobileNet, ResNet, model summary | Notebook Section 9, 10 |
| Huấn luyện | FIG-TRAIN | Training loss/accuracy của MobileNet và ResNet | `training_artifacts/figures` |
| Đánh giá | FIG-EVAL | Confusion matrix và so sánh metric | `model_comparison.csv`, `training_artifacts/figures` |
| Dự đoán mẫu | FIG-PRED | Ảnh dự đoán đúng/sai của từng mô hình | Notebook Section 11 |

Việc đánh mã hình giúp báo cáo giữ cấu trúc nhất quán. Khi rà soát báo cáo, nhóm chỉ cần kiểm tra từng mã hình đã có caption, đã được nhắc trong nội dung và đã khớp với kết quả thực nghiệm tương ứng.

## Phụ lục B. Các flow Mermaid của dự án

Nhóm xây dựng thêm các flow Mermaid để mô tả pipeline ở mức trực quan. Các flow này hỗ trợ phần trình bày quy trình trong báo cáo và có thể dùng lại trong slide bảo vệ.

| File Mermaid | Nội dung |
|---|---|
| `project_overall_flow.mmd` | Flow tổng quan từ crawl dữ liệu đến lưu artifact sau đánh giá |
| `crawl_pipeline_flow.mmd` | Flow chi tiết cho bước đọc config, chọn keyword, crawl Naver/DuckDuckGo và lưu metadata |
| `clean_dedup_flow.mmd` | Flow làm sạch ảnh, chuẩn hóa JPEG, tính perceptual hash và lọc duplicate |
| `train_eval_flow.mmd` | Flow từ `data/cleaned` đến split, preprocessing, train MobileNet/ResNet và evaluate |
| `kaggle_training_flow.mmd` | Flow chuẩn bị dữ liệu Kaggle, chạy notebook và tải artifact |

Các flow này làm rõ tính lặp của pipeline. Đặc biệt, giai đoạn crawl không chạy một lần duy nhất mà được kiểm tra theo số lượng ảnh còn thiếu ở từng label. Giai đoạn làm sạch cũng không chỉ xóa ảnh lỗi, mà còn chuẩn hóa ảnh và tạo dữ liệu đầu vào nhất quán cho quá trình train.

## Phụ lục C. Artifact thực nghiệm

Notebook lưu các artifact chính trong thư mục `training_artifacts`. Các artifact này giúp nhóm tái kiểm tra mô hình, lấy số liệu đưa vào báo cáo và phân tích dự đoán trên test set.

| Artifact | Ý nghĩa |
|---|---|
| `models/mobilenet_best.keras` | MobileNet tốt nhất theo validation accuracy |
| `models/resnet_best.keras` | ResNet tốt nhất theo validation accuracy |
| `models/mobilenet_final.keras` | MobileNet tại trạng thái cuối sau train |
| `models/resnet_final.keras` | ResNet tại trạng thái cuối sau train |
| `mobilenet_history.csv` | Lịch sử loss/accuracy của MobileNet |
| `resnet_history.csv` | Lịch sử loss/accuracy của ResNet |
| `model_comparison.csv` | Bảng so sánh Accuracy, Precision, Recall, F1-score, số tham số, thời gian train |
| `*_classification_report.csv` | Báo cáo precision/recall/F1-score theo từng class |
| `*_test_predictions.csv` | Dự đoán của mô hình trên từng ảnh test |
| `*_test_probabilities.npy` | Xác suất dự đoán của mô hình trên test set |

Các file prediction CSV có vai trò quan trọng trong phần phân tích định tính. Từ các file này, nhóm xác định được ảnh nào dự đoán đúng, ảnh nào dự đoán sai, class thật là gì, class dự đoán là gì và độ tự tin của mô hình.

## Phụ lục D. Rà soát nội dung báo cáo

Báo cáo được rà soát theo các tiêu chí sau:

| Nhóm tiêu chí | Nội dung rà soát |
|---|---|
| Cấu trúc | Có đủ tóm tắt, bảng phân công, mục lục, 6 phần nội dung chính và phụ lục |
| Dữ liệu | Có mô tả nguồn crawl, keyword, cấu trúc folder, số lượng mẫu và thống kê kích thước ảnh |
| Làm sạch | Có trình bày ảnh lỗi, ảnh quá nhỏ, chuẩn hóa RGB JPEG, rename và lọc duplicate |
| Đặc trưng | Có trình bày resize, normalize, augmentation, PCA/t-SNE và phân tích trước train |
| Mô hình | Có lý thuyết MobileNet, ResNet, cấu trúc layer, loss, optimizer, learning rate và callback |
| Đánh giá | Có Accuracy, Precision, Recall, F1-score, confusion matrix, classification report và ảnh đúng/sai |
| So sánh | Có phân tích trade-off giữa MobileNet và ResNet về chất lượng và chi phí tính toán |
| Hình thức | Hình có caption, bảng có tiêu đề, số liệu khớp với CSV và nội dung không mâu thuẫn |

## Phụ lục E. Bảng tóm tắt đóng góp của nhóm

| Thành viên | Đóng góp chính | Sản phẩm liên quan |
|---|---|---|
| Nguyễn Thanh Hiếu | Xây dựng pipeline AI, MobileNet, xử lý notebook, lưu artifact | Notebook train/evaluate, phần MobileNet, biểu đồ kết quả |
| Nguyễn Mạnh Kiên | Xây dựng ResNet, đánh giá test set, so sánh mô hình | ResNet, confusion matrix, classification report, bảng metric |
| Nguyễn Văn Tiến | Crawl dữ liệu, theo dõi nguồn ảnh, hỗ trợ lọc dữ liệu | `data/raw`, log crawl, thống kê tiến độ |
| Huỳnh Ngọc Khánh Linh | Tổ chức dữ liệu, hỗ trợ crawl, chuẩn bị dataset Kaggle | `data/cleaned`, cấu trúc folder, kiểm tra dữ liệu |
| Cả nhóm | Lọc thủ công, kiểm tra nhãn, tổng hợp báo cáo | Dữ liệu cuối, hình minh họa, báo cáo hoàn chỉnh |

Phần đóng góp được phân chia theo hướng mỗi thành viên phụ trách một mảng chính nhưng vẫn có sự kiểm tra chéo. Cách làm này giúp giảm rủi ro sai nhãn ở dữ liệu và giảm sai sót khi đưa kết quả mô hình vào báo cáo.


