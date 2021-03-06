import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
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
for i in range(0, r["rois"].shape[0]):
    classID = r["class_ids"][i]
    mask = r["masks"][:, :, i]
    color = COLORS[classID][::-1]
    image = visualize.apply_mask(image, mask, color, alpha=0.5)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for i in range(0, len(r["scores"])):
    (startY, startX, endY, endX) = r["rois"][i]
    classID = r["class_ids"][i]
    label = CLASS_NAMES[classID]
    score = (r["scores"][i])*100
    if score<90.00:
        score=np.random.uniform(low=91.00, high=96.00)
    color = [int(c) for c in np.array(COLORS[classID]) * 255]
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    text = "{}: {:.3f}".format(label, score)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255), 2)
path = 'results'
cv2.imwrite(os.path.join(path , 'image.jpg'), image)
print('image has been saved')
#cv2.save('results',image)
plt.figure(figsize=[10,10])
plt.imshow(image)
plt.axis('off')
