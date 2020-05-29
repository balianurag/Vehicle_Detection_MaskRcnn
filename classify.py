import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
from skimage import io, transform
import colorsys,argparse , imutils, random, cv2, os
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,help="path to class labels file")
ap.add_argument("-i", "--image", required=True,help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)
class SimpleConfig(Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
config = SimpleConfig()
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)
print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]
labellist=[]
masklist=[]
vehicle=[]
for i in range(0, r["rois"].shape[0]):
    classID = r["class_ids"][i]
    mask = r["masks"][:, :, i]
    masklist.append(masklist)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for i in range(0, len(r["scores"])):
    (startY, startX, endY, endX) = r["rois"][i]
    classID = r["class_ids"][i]
    label = CLASS_NAMES[classID]
    labellist.append(label)
    score = r["scores"][i]
    color = [int(c) for c in np.array(COLORS[classID]) * 255]
    text = "{}: {:.3f}".format(label, score)
    y = startY - 10 if startY - 10 > 10 else startY + 10
#if(len(masklist)!=0):
#    cv2.putText(image,"Vehicle is present in frame",(30,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 2)
for i in labellist:
    if (i=='car'or i=='bus'or i=='truck'):
        vehicle.append(1)
    else:
        continue
if any(vehicle)==True:
    cv2.putText(image,"Vehicle is present in frame",(30,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 2)
else:
    cv2.putText(image,"Vehicle is not present in frame",(30,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 2)
path = 'classification_results'
cv2.imwrite(os.path.join(path , 'image.jpg'), image)
print('image has been saved')
plt.figure(figsize=[10,10])
plt.imshow(image)
plt.axis('off')
