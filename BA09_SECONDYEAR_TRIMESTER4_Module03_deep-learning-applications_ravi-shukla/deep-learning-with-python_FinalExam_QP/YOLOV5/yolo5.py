# -*- coding: utf-8 -*-
"""yolo5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w7_Wv7kFgMPP4tDAO15dmoXkaTWGBjD5
"""

!git clone https://github.com/ultralytics/yolov5

!cd '/content/yolov5'

!pip install -r '/content/yolov5/requirements.txt'

!unzip -qq '/content/archive.zip'

!python '/content/yolov5/train.py' --imgsz 640 --batch-size 16 --epochs 15 --data '/content/yolov5/data/data.yaml' --weights yolov5s.pt

!python '/content/yolov5/detect.py' --weights '/content/yolov5/runs/train/exp/weights/best.pt' --source '/content/test/images/11739075086_a05dcf50b9_c_jpeg.rf.a867923221d1693ed1ede5546fbdc53b.jpg'

!python '/content/yolov5/detect.py' --weights '/content/yolov5/runs/train/exp/weights/best.pt' --source '/content/test/images/24642486142_7cb8f558b6_c_jpeg.rf.bfe9c3d0187c0894075cd678dacf11fa.jpg'

!python '/content/yolov5/detect.py' --weights '/content/yolov5/runs/train/exp/weights/best.pt' --source '/content/test/images/2679486770_050f0e92a8_c_jpeg.rf.45b7b7c6bf1f7968bf7be7b83fab3da2.jpg'

!python '/content/yolov5/detect.py' --weights '/content/yolov5/runs/train/exp/weights/best.pt' --source '/content/test/images/6245613788_dd7b25ee8c_c_jpeg.rf.998f5996ee0596ebe79591b2f31ecfcd.jpg'



!python '/content/yolov5/detect.py' --weights '/content/yolov5/runs/train/exp/weights/best.pt' --source '/content/test/images/9273411600_25edab279d_c_jpeg.rf.dd0cf9b496b5bb967b8085515058dfe3.jpg'

!jupyter nbconvert --to html  yolov5.ipynb

