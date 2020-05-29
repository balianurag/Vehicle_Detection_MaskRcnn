# Vehicle_Detection_MaskRcnn

1) For testing object Detection run below given command :-
python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image image_dir_path/image_name.jpg

2) For classifying the frame on the basis of the vehicle detected or not in the frame run below given command :-
python classify.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image image_dir_path/image_name.jpg

3) Download the mask_rcnn_coco.h5 from below given link address :-
https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
