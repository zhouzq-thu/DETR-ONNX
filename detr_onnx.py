import cv2 as cv
import numpy as np
from scipy.special import softmax
import onnxruntime as ort

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)

class DetrONNX:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.input_shape = self.sess.get_inputs()[0].shape
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.stride = int(meta.get('stride', -1))
        self.class_names = eval(meta.get('names', '{}'))
    
    def detect(self, image: np.ndarray, prob_threshold = 0.7):
        N, C, H, W = self.input_shape
        img = image
        if isinstance(H, str) or isinstance(W, str):
            if self.stride > 0:
                H = (image.shape[0] + self.stride - 1) // self.stride * self.stride
                W = (image.shape[1] + self.stride - 1) // self.stride * self.stride
                img = cv.resize(image, (H, W))
        else:
            img = cv.resize(image, (H, W))
        
        img = img[..., ::-1] # BGR to RGB
        img = img[None].astype(np.float32) / 255
        img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))
        
        ort_inputs = {self.input_name: img}
        scores, boxes = self.sess.run(None, ort_inputs)
        
        probas = softmax(np.array(scores), axis=-1)[0, :, :-1]
        keep = np.max(probas, axis=-1) > prob_threshold
        bboxes_xyxy = rescale_bboxes(boxes[0, keep], image.shape[:2][::-1])
        return probas[keep], bboxes_xyxy
    
    def plot_result(self, image: np.ndarray, prob: np.ndarray, boxes: np.ndarray):
        img = image.copy()
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
            cl = p.argmax()
            name = self.class_names[cl] if cl in self.class_names else str(cl)
            label_text = '{}: {:.2f}%'.format(name, p[cl] * 100)
            plot_one_box((xmin, ymin, xmax, ymax), img, label=label_text)
        return img

if __name__ == "__main__":
    model_path = "models/rt-detrv2.onnx"
    detr = DetrONNX(model_path)
    
    img = cv.imread("images/kite.jpg")
    prob, boxes = detr.detect(img)
    res = detr.plot_result(img, prob, boxes)
    
    cv.imshow("res", res)
    cv.waitKey(0)
    