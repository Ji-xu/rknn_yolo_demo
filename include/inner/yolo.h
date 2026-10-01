#ifndef __YOLO_H__
#define __YOLO_H__

#include <string>
#include <vector>
#include <memory>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

struct Detection {
    float x;
    float y;
    float w;
    float h;
    int class_id;
    float score;
};

class Yolo {
public:
    Yolo(const std::string& model_path, float confidence_threshold = 0.25, float nms_threshold = 0.45)
    : model_path_(model_path), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold) {}
    ~Yolo();

    int init();
    int infer(const cv::Mat& image, std::vector<Detection>& detections);




private:
    /* 成员函数 */
    int pre_process();
    int post_process();
    int nms();

    /* 成员变量 */
    std::string model_path_;
    float confidence_threshold_;
    float nms_threshold_;
};

#endif // __YOLO_H__