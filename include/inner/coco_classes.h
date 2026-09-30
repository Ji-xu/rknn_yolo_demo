#ifndef COCO_CLASSES_H
#define COCO_CLASSES_H

#include <array>

// YOLOv8 COCO 80 类，索引 0~79 与模型输出 class_id 一一对应
static const std::array<const char*, 80> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// 安全访问：越界返回 "unknown"
inline const char* coco_cls_name(int id) {
    if (id >= 0 && id < static_cast<int>(COCO_CLASSES.size())) {
        return COCO_CLASSES[id];
    }
    return "unknown";
}

#endif // COCO_CLASSES_H