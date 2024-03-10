import numpy as np

import torch

from PIL import Image
from sld.utils import nms, post_process
from utils.utils import free_memory
from utils.parse import p


def check_same_object(box1, box2, iou_threshold=0.9):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Check if there is no intersection (one or both boxes have zero area)
    if w_intersection <= 0 or h_intersection <= 0:
        return False

    # Calculate the area of intersection
    intersection_area = w_intersection * h_intersection

    # Calculate the areas of the two bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the Union area (area of box1 + area of box2 - intersection area)
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    # print("IoU = ", iou, flush=True)
    # print(iou)
    if iou > iou_threshold:
        return True
    else:
        return False


def pop_entry_via_name(obj_name, det_list):
    for idx, obj in enumerate(det_list):
        if obj_name == obj[0]:
            ret = det_list[idx]
            det_list.pop(idx)
            return ret
    return None


def pop_entry_via_box(bbox, det_list):
    for idx, obj in enumerate(det_list):
        if bbox[0] == obj[0] and check_same_object(bbox[1], obj[1]) is True:
            ret = det_list[idx]
            det_list.pop(idx)
            return ret
    return None


def peak_bbox_via_name(target_base_name, det_results):
    return_list = []
    for obj in det_results:
        base_name = obj[0].split(" #")[0]
        if base_name == target_base_name:
            return_list.append(obj)
    return return_list


def class_aware_nms(
    bounding_boxes, confidence_score, labels, threshold, input_in_pixels=False
):
    """
    This NMS processes boxes of each label individually.
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    picked_boxes, picked_score, picked_labels = [], [], []

    labels_unique = np.unique(labels)
    for label in labels_unique:
        bounding_boxes_label = [
            bounding_box
            for i, bounding_box in enumerate(bounding_boxes)
            if labels[i] == label
        ]
        confidence_score_label = [
            confidence_score_item
            for i, confidence_score_item in enumerate(confidence_score)
            if labels[i] == label
        ]
        labels_label = [label] * len(bounding_boxes_label)
        picked_boxes_label, picked_score_label, picked_labels_label = nms(
            bounding_boxes_label,
            confidence_score_label,
            labels_label,
            threshold=threshold,
            input_in_pixels=input_in_pixels,
            return_array=False,
        )
        picked_boxes += picked_boxes_label
        picked_score += picked_score_label
        picked_labels += picked_labels_label

    picked_boxes, picked_score, picked_labels = (
        np.array(picked_boxes),
        np.array(picked_score),
        np.array(picked_labels),
    )

    return picked_boxes, picked_score, picked_labels


class Detector:
    def __init__(self):
        # Initialize class variables
        self.prompt = None
        self.object_lists = []
        self.primitive_count_dict = {}
        self.attribute_count_dict = {}
        self.pred_primitive_count_dict = {}
        self.pred_attribute_count_dict = {}

    def register_objects(self, prompt, object_lists):
        """
        Register objects and their attributes from the given object lists.
        """
        # Reset class variables
        self.prompt = prompt
        self.object_lists = object_lists
        self.primitive_count_dict = {}
        self.attribute_count_dict = {}
        self.pred_primitive_count_dict = {}
        self.pred_attribute_count_dict = {}

        for item in object_lists:
            key, values = item
            # TODO: Some words might cause undesired results
            #       Will change it to NLTK to enhance robustness
            # if p.singular_noun(key) is not False:
            #     key = p.singular_noun(key)
            self.pred_primitive_count_dict[key] = 0
            ask_attribute = False
            for value in values:
                if value is not None:
                    ask_attribute = True
                    real_key = f"{value} {key}"
                    self.attribute_count_dict[real_key] = (
                        self.attribute_count_dict.get(real_key, 0) + 1
                    )
                    self.pred_attribute_count_dict[real_key] = 0
            if ask_attribute is False:
                self.primitive_count_dict[key] = len(values)
            else:
                self.primitive_count_dict[key] = -1
        # print(self.primitive_count_dict)  # {'princess': 2, 'dwarf': 1}
        # print(self.attribute_count_dict)  # {('princess', 'pink'): 2, ('dwarf', 'blue'): 1}

    def run(self, **kwargs):
        return NotImplementedError

    def detect(self, **kwargs):
        return NotImplementedError

    def parse_list(self, det_results, llm_suggestions, iou_threshold=0.9):
        """
        Take detection result and llm suggestions (two lists) as input,
        prase them into four categories: add / move / remove / change attr
        """
        key_curr = set([obj[0] for obj in det_results])
        key_goal = set([obj[0] for obj in llm_suggestions])
        add_keys = key_goal - key_curr  # Add / Change Attr
        sub_keys = key_curr - key_goal  # Remove / Change Attr
        same_keys = key_curr.intersection(key_goal)  # Possible Move

        remove_objects = []
        add_objects = []
        move_objects = []
        change_attr_object = []
        preserve_objects = []
        # Move or unchanged
        for key in same_keys:
            old_entry = pop_entry_via_name(key, det_results)
            new_entry = pop_entry_via_name(key, llm_suggestions)
            if (
                check_same_object(old_entry[1], new_entry[1], iou_threshold) is False
            ):  # Move
                move_objects.append((tuple(old_entry), tuple(new_entry)))
            else:
                preserve_objects.append(tuple(old_entry))

        # Add or change attribute
        for key in add_keys:
            new_entry = pop_entry_via_name(key, llm_suggestions)
            base_object = key.split(" #")[0].split(" ")[-1]
            change_attr = False
            # Peak objects with basename
            candidates = peak_bbox_via_name(base_object, det_results)
            for obj in candidates:
                if check_same_object(obj[1], new_entry[1], iou_threshold):
                    change_attr = True
                    change_attr_object.append(tuple(new_entry))
                    # also remove it from det_attr_dict if needed
                    sub_keys.remove(obj[0])
                    break
            # Still need to add new object
            if change_attr is False:
                add_objects.append(tuple(new_entry))
        # Removal Part
        for key in sub_keys:
            entry = pop_entry_via_name(key, det_results)
            remove_objects.append(tuple(entry))

        # Check attribute change
        return preserve_objects, remove_objects, add_objects, move_objects, change_attr_object

    def summarize_result(self, attribute_objects, primitive_objects):
        """
        Summarizes the result of attribute and primitive objects.

        Args:
            attribute_objects (dict): A dictionary of attribute objects.
            primitive_objects (dict): A dictionary of primitive objects.

        Returns:
            list: A list of tuples representing the final result, where each tuple contains the object name and its corresponding bbox.
        """
        # walk through attribute dict
        attribute_results = {}
        primitive_results = {}
        for key in self.attribute_count_dict:
            non_attr_key = key.split()[-1]
            for _ in range(self.attribute_count_dict[key]):
                ret = pop_entry_via_name(key, attribute_objects)
                if ret is not None:
                    attribute_results[ret[0]] = attribute_results.get(ret[0], []) + [
                        ret[1]
                    ]
                    primitive_query = [non_attr_key, ret[1]]
                    pop_entry_via_box(primitive_query, primitive_objects)
                    self.pred_attribute_count_dict[key] += 1
                    self.pred_primitive_count_dict[non_attr_key] += 1
                else:
                    break
        # walk through non-attribute dict
        for key in self.primitive_count_dict:
            # if self.primitive_count_dict[key] == -1:

            while True:
                ret = pop_entry_via_name(key, primitive_objects)
                if ret is not None:
                    primitive_results[ret[0]] = primitive_results.get(ret[0], []) + [
                        ret[1]
                    ]
                    self.pred_primitive_count_dict[key] += 1
                else:
                    break

        det_objects_counter = {}
        final_result = []
        for key in self.attribute_count_dict:
            if key in attribute_results:
                N = len(attribute_results[key])
            else:
                N = 0
            for i in range(N):
                det_objects_counter[key] = det_objects_counter.get(key, 0) + 1
                obj_id = det_objects_counter[key]
                final_result.append((f"{key} #{obj_id}", attribute_results[key][i]))
            base_name = key.split()[-1]
            start_idx = N
            if base_name in primitive_results:
                N_base = len(primitive_results[base_name])
            else:
                N_base = 0

            end_idx = min(self.attribute_count_dict[key], N + N_base)
            for i in range(start_idx, end_idx):
                det_objects_counter[base_name] = (
                    det_objects_counter.get(base_name, 0) + 1
                )
                obj_id = det_objects_counter[base_name]
                final_result.append(
                    (f"{base_name} #{obj_id}", primitive_results[base_name][i])
                )

        # non attribute one
        for key in self.primitive_count_dict:
            # if self.primitive_count_dict[key] == -1:
            #     continue  # pass attribute binding cases
            if key in primitive_results:
                N = len(primitive_results[key])
            else:
                N = 0

            # check if it is an attribute binding case
            # attr_binding = False
            for attr_key in self.attribute_count_dict:
                ret = attr_key.split(" ")
                if len(ret) == 2 and ret[1] == key:
                    # attr_binding = True
                    N = min(
                        self.attribute_count_dict[attr_key]
                        - len(attribute_results.get(attr_key, [])),
                        0,
                    )
                    break

            det_objects_counter[key] = det_objects_counter.get(key, 0)
            for obj_id in range(det_objects_counter[key], N):
                final_result.append(
                    (f"{key} #{obj_id + 1}", primitive_results[key][obj_id])
                )
                det_objects_counter[key] + 1
        return final_result

class OWLVITV1Detector(Detector):
    def __init__(self):
        super().__init__()
        # Initialize object detector

        # load jax models
        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        owl_vit_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = owl_vit_model.eval().to("cuda")

    def detect(self, img_path, split, score_threshold, nms_threshold=0.15):
        # data preprocessing
        if split == "attribute":
            target_objects = [x for x in self.attribute_count_dict]
        elif split == "primitive":
            target_objects = [x for x in self.primitive_count_dict]
        if len(target_objects) == 0:
            return []
        image = Image.open(img_path)
        texts = [[f"image of {p.a(obj)}" for obj in target_objects]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            outputs = self.model(**inputs)

        width, height = image.size

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([[height, width]])
        target_sizes = target_sizes.cuda()
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0
        )
        assert len(results) == 1
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        # Hack: Assume width = height
        boxes = torch.clip(boxes, 0, width)
        boxes = boxes.cpu().detach()
        # xyxy ranging from 0 to 1
        boxes = np.array(
            [
                [x_min / width, y_min / height, x_max / width, y_max / height]
                for (x_min, y_min, x_max, y_max), score in zip(boxes, scores)
                if score >= score_threshold
            ]
        )
        labels = np.array(
            [
                label.cpu().numpy()
                for label, score in zip(labels, scores)
                if score >= score_threshold
            ]
        )
        scores = np.array(
            [
                score.cpu().detach().numpy()
                for score in scores
                if score >= score_threshold
            ]
        )
        boxes, scores, labels = class_aware_nms(boxes, scores, labels, nms_threshold)
        ret_results = []
        for box, score, label in zip(boxes, scores, labels):
            box = box.tolist()
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]  # xyxy to xyhw
            box = post_process(box)

            # print(
            #     f"Detected {target_objects[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}"
            # )
            ret_results.append(
                (
                    f"{target_objects[label]}",
                    box,
                )
            )

        return ret_results

    def run(self, prompt, object_lists, img_path):
        self.register_objects(prompt, object_lists)
        attribute_objects = self.detect(
            img_path, split="attribute", score_threshold=0.2
        )
        primitive_objects = self.detect(
            img_path, split="primitive", score_threshold=0.15
        )
        free_memory()
        print(f"* attr object: {attribute_objects}")
        print(f"* prim object: {primitive_objects}")
        return self.summarize_result(attribute_objects, primitive_objects)


class OWLVITV2Detector(Detector):
    def __init__(self, attr_detection_threshold=0.6, prim_detection_threshold=0.2, nms_threshold=0.5):
        super().__init__()
        # Initialize object detector
        # set score threshold here
        self.default_attr_detection_threshold = attr_detection_threshold
        self.default_prim_detection_threshold = prim_detection_threshold
        self.default_nms_threshold = nms_threshold

        # load jax models
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        owl_vit_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = owl_vit_model.eval().to("cuda")

    def detect(self, img_path, split, score_threshold, nms_threshold):
        # data preprocessing
        if split == "attribute":
            target_objects = [x for x in self.attribute_count_dict]
        elif split == "primitive":
            target_objects = [x for x in self.primitive_count_dict]
        if len(target_objects) == 0:
            return []
        image = Image.open(img_path)
        texts = [[f"image of {p.a(obj)}" for obj in target_objects]]

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            outputs = self.model(**inputs)

        width, height = image.size

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([[height, width]])
        target_sizes = target_sizes.cuda()
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0
        )
        assert len(results) == 1
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        # Hack: Assume width = height
        boxes = torch.clip(boxes, 0, width)
        boxes = boxes.cpu().detach()
        # xyxy ranging from 0 to 1
        boxes = np.array(
            [
                [x_min / width, y_min / height, x_max / width, y_max / height]
                for (x_min, y_min, x_max, y_max), score in zip(boxes, scores)
                if score >= score_threshold
            ]
        )
        labels = np.array(
            [
                label.cpu().numpy()
                for label, score in zip(labels, scores)
                if score >= score_threshold
            ]
        )
        scores = np.array(
            [
                score.cpu().detach().numpy()
                for score in scores
                if score >= score_threshold
            ]
        )
        boxes, scores, labels = class_aware_nms(boxes, scores, labels, nms_threshold)
        ret_results = []
        for box, score, label in zip(boxes, scores, labels):
            box = box.tolist()
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]  # xyxy to xyhw
            box = post_process(box)

            # print(
            #     f"Detected {target_objects[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}"
            # )
            ret_results.append(
                (
                    f"{target_objects[label]}",
                    box,
                )
            )

        return ret_results

    def run(self, prompt, object_lists, img_path, **kwargs):
        attr_detection_threshold = kwargs.get("attr_detection_threshold", self.default_attr_detection_threshold)
        prim_detection_threshold = kwargs.get("prim_detection_threshold", self.default_prim_detection_threshold)
        nms_threshold = kwargs.get("nms_threshold", self.default_nms_threshold)
        self.register_objects(prompt, object_lists)
        attribute_objects = self.detect(
            img_path, split="attribute", score_threshold=attr_detection_threshold, nms_threshold=nms_threshold,
        )
        primitive_objects = self.detect(
            img_path, split="primitive", score_threshold=prim_detection_threshold, nms_threshold=nms_threshold,
        )
        free_memory()
        print(f"* attr object: {attribute_objects}")
        print(f"* prim object: {primitive_objects}")
        return self.summarize_result(attribute_objects, primitive_objects)


if __name__ == "__main__":
    prompt = "A realistic photo with a monkey sitting above a green motorcycle on the left and another raccoon sitting above a blue motorcycle on the right"
    objects = [
        ["monkey", [None]],
        ["motorcycle", ["green", "blue"]],
        ["raccoon", [None]],
    ]
    det = OWLVITV2Detector()
    det_results = det.run(prompt, objects, "../demo/dalle3_motor.png")
    llm_suggestions = [
        ["monkey #1", [0.009, 0.006, 0.481, 0.821]],
        ["green motorcycle #1", [0.016, 0.329, 0.506, 0.6]],
        ["blue motorcycle #1", [0.516, 0.329, 0.484, 0.6]],
        ["raccoon #1", [0.46, 0.123, 0.526, 0.62]],
    ]
    remove_objects, add_objects, move_objects, change_attr_object = det.parse_list(
        det_results, llm_suggestions
    )
    print(
        f"- Remove: {remove_objects}\n"
        f"- Add: {add_objects}\n"
        f"- Move: {move_objects}\n"
        f"- Change-Attr: {change_attr_object}\n"
    )
