# real_time_fall_detection

!!! More details will be added after the publication !!!

Papers based on:
Fallen People Detection Capabilities Using Assistive Robot ( https://www.mdpi.com/2079-9292/8/9/915 ), 
RGB camera-based fallen person detection system embedded on a mobile platform ( https://www.sciencedirect.com/science/article/pii/S0957417422001890 )

Used datasets:
COCO: https://cocodataset.org/#download
E-FDPS: https://gram.web.uah.es/data/datasets/fpds/index.html
LE2I (imvia): https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html

YOLOv8 is applied (https://github.com/ultralytics/ultralytics) and the installation instructions should first be followed before the installation of requirements.txt .

For the target dataset <LE2I (imvia): https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html> run the unzip.py script in the folder targer_dataset. In fall_detection_DL.py at 95 serie give the path of the dataset folder that is created and in 96 the path of labels.csv .
