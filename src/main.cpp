#include <cstdio>
#include <opencv2/opencv.hpp>
#include "inner/yolo.h"
#include "inner/coco_classes.h"

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <model.rknn> <image>\n", argv[0]);
        return -1;
    }
    /* 读图 */
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        printf("Failed to read image: %s\n", argv[2]);
        return -1;
    }
    /* 创建 yolo 对象并初始化 */
    Yolo yolo(argv[1]);
    if (yolo.init() != 0) {
        return -1;
    }
    /* 推理 */
    std::vector<Detection> detections;
    if (yolo.infer(image, detections) != 0) {
        return -1;
    }
    /* 画框 */
    for (const auto& d : detections) {
        cv::Rect box((int)d.x, (int)d.y, (int)d.w, (int)d.h);
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", coco_cls_name(d.class_id), d.score);
        cv::putText(image, label, {box.x, box.y - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    cv::imwrite("result.jpg", image);
    printf("Detected %d objects -> result.jpg\n", (int)detections.size());
    return 0;
}
