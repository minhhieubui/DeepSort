### Create file setup.txt to pip install library
torch
utils
imutils
Pillow
deep-sort-realtime

### Create file data.ext in Yolo V9 and download file " classes.name (COCO data) , highway.mp4 , test . mp4" 

### Create file object_tracking to code MOT and DeepSort 
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape      # detect va tra ket qua trong code, ma khong can phai ghi trong file exp 
# from ultralytics import YOLO
#
# from yolov9 import detect
# from yolov9.utils.plots import colors

# Config value
video_path = "data.ext/test.mp4"
conf_threshold = 0.5      
tracking_class = None   # None : track all ,  muốn tracking car ( ID = 2 trong bộ data COCO)

# Khởi tạo Deepsort
tracker = DeepSort(max_age=5) # sau 5 lần tracking kh tìm ra vật thể thì xóa vật thể khỏi bộ nhớ

# Khởi tạo yolov9
device = "cpu" #"cuda" : GPU , "cpu" : CPU, "mps:0"
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt",device=device, fuse = True)  # fuse = True : cho biết rằng mô hình được tối ưu hóa chạy trên CPU or GPU
model = AutoShape(model)     # Model sẽ bao gồm cả cấu trúc mạng và trọng số của YOLO V9 để cbi dự đoán và detect ảnh hoặc video 

# Load classname từ file classes.names
with open("data.ext/classes.names") as f:
    class_names = f.read().strip().split('\n')      # đọc cả file, bỏ hêt dấu cách, ký tự đầu cuối, tách ra bằng \n để xuống dòng

colors = np.random.randint(0,255, size=(len(class_names),3))  # tạo mảng color, mỗi lớp có mỗi màu khác nhau
tracks =[]   # Khởi tạo vật thể track được bằng rỗng

# Khởi tạo VideoCapture để đọc file từ video
cap = cv2.VideoCapture(video_path)
# Thay đổi độ phân giải (ví dụ: 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Tiến hành đọc từng frame từ video
frame_count = 0
while True:
    # Đọc
 ret, frame = cap.read()
 if not ret:
    continue

 frame_count += 1
 if frame_count % 5 != 0 :
    # Đưa qua model để detect
    results = model(frame)
    detect = []
    for detect_object in results.pred[0]:   # khi predict model ở dạng 1 mảng có 1 phần tử, pred[0] giúp lấy phần tử đầu tiên
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]  # yolov9 predict return value : vị trí của class ID với độ tin cây, và tọa độ là bao nhiêu
        x1, y1, x2, y2 = map(int,bbox)  # trả ra tọa độ thực
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue
        
        # Duyet het tat ca, xet object > 0.5 và đúng với class
        detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])  # thêm phần tử vào cuối mảng detect với bbox, conf và class_id


    # Cập nhật, gán ID bằng Deepsort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:                # mỗi track đại diêện cho mỗi object đang được theo dõi
        if track.is_confirmed():      # Được xác nhận bởi tracker thì mới làm 
            track_id = track.track_id

            # Lấy tọa độ class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()          # Chuyển tọa độ sang dạng left, top, right, bottom
            class_id = track.get_det_class()    # Lấy ID của lớp đã phát hiện
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]            # Lấy màu tương ứng các class ID
            B, G, R = map(int, color)           # Chuyển đổi màu sang định dạng BGR cho OpenCV

            label = "{}-{}".format(class_names[class_id], track_id)    # label gồm tên lớp car-2 , ID theo dõi (track ID)

            # Vẽ HCN cho object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)

            # Vẽ nền cho label
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label)*12, y1), (B, G, R), -1)

            # Vẽ nhãn lên ảnh
            cv2.putText(frame, label, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # out.write(frame)

    # Show hình ảnh lên màn hình
    cv2.namedWindow("OT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OT", 800, 600)
    cv2.imshow("OT", frame)


    # Bấm q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()





