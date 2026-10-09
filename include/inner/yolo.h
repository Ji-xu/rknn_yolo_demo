#ifndef __YOLO_H__
#define __YOLO_H__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

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

    Yolo(const Yolo&) = delete; //禁止Yolo b = a; 或 Yolo b(a);
    Yolo& operator=(const Yolo&) = delete; //禁止 b = a;

    int init();
    int infer(const cv::Mat& image, std::vector<Detection>& detections);

private:
    /* 成员函数 */
    void preprocess(const cv::Mat& bgr_image, cv::Mat& preprocess_image, float& scale, int& pad_h, int& pad_w);
    int postprocess(const std::vector<rknn_output>& outputs, float scale, int pad_h, int pad_w, std::vector<Detection>& results);
    void nms(std::vector<Detection>& dets);

    /* 成员变量 */
    std::string model_path_;
    float confidence_threshold_;
    float nms_threshold_;
    rknn_context ctx_ = 0;   // 推理句柄
    int input_w_ = 0;
    int input_h_ = 0;
    int n_output_ = 0;
    std::vector<rknn_tensor_attr> output_tensor_attrs_;
};

#endif // __YOLO_H__