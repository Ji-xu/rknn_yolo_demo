#include <cmath>
#include "inner/yolo.h"

Yolo::~Yolo() {
    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

/* 加载模型并查询输入信息 */
int Yolo::init(){
    //读取模型文件
    FILE* model_file = fopen(model_path_.c_str(), "rb");
    if (!model_file) {
        printf("Failed to open model file: %s\n", model_path_.c_str());
        return -1;
    }
    fseek(model_file, 0, SEEK_END);
    size_t model_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    std::vector<char> model_data(model_size);
    fread(model_data.data(), 1, model_size, model_file);
    fclose(model_file);
    //初始化RKNN上下文
    int ret = rknn_init(&ctx_, model_data.data(), model_size, 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("Failed to initialize RKNN context: %d\n", ret);
        return -1;
    }
    /* 查询输入输出数量 */
    rknn_input_output_num input_output_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &input_output_num, sizeof(input_output_num));
    if (ret != RKNN_SUCC) {
        printf("Failed to query input number: %d\n", ret);
        return -1;
    }
    printf("Input number: %d, Output number: %d\n", input_output_num.n_input, input_output_num.n_output);
    /* 查询输入张量属性 */
    rknn_tensor_attr input_tensor_attr = {};
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_tensor_attr, sizeof(input_tensor_attr));
    if (ret != RKNN_SUCC) {
        printf("Failed to query input tensor attribute: %d\n", ret);
        return -1;
    }
    /* dims = [n, h, w, c]（NHWC） */
    input_h_ = input_tensor_attr.dims[1];
    input_w_ = input_tensor_attr.dims[2];
    /* 查询输出张量属性 */
    n_output_ = input_output_num.n_output;
    output_tensor_attrs_.resize(n_output_);
    for (int i = 0; i < n_output_; ++i) {
        output_tensor_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_tensor_attrs_[i], sizeof(output_tensor_attrs_[i]));
        if (ret != RKNN_SUCC) {
            printf("Failed to query output tensor attribute %d: %d\n", i, ret);
            return -1;
        }
    }
    return 0;
}


int Yolo::infer(const cv::Mat& image, std::vector<Detection>& detections){
    detections.clear();
    /* 预处理图像 */
    cv::Mat preprocess_image;
    float scale;
    int pad_h, pad_w;
    preprocess(image, preprocess_image, scale, pad_h, pad_w);
    /* 设置输入 */
    rknn_input input = {};
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;   // cv::Mat 是 HWC 排布
    input.size = preprocess_image.total() * preprocess_image.elemSize();
    input.buf = preprocess_image.data;
    int ret = rknn_inputs_set(ctx_, 1, &input);
    if (ret != RKNN_SUCC) {
        printf("Failed to set input: %d\n", ret);
        return -1;
    }
    /* 执行推理 */
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        printf("Failed to run inference: %d\n", ret);
        return -1;
    }
    /* 后处理 */
    std::vector<rknn_output> outputs(n_output_);
    for (int i = 0; i < n_output_; ++i) {
        outputs[i].want_float = 1;   // 拿反量化后的 float，postprocess 直接用
    }
    ret = rknn_outputs_get(ctx_, n_output_, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        printf("Failed to get outputs: %d\n", ret);
        return -1;
    }
    ret = postprocess(outputs, scale, pad_h, pad_w, detections);
    /* 释放输出 */
    rknn_outputs_release(ctx_, n_output_, outputs.data());

    return ret;
}


void Yolo::preprocess(const cv::Mat& bgr_image, cv::Mat& preprocess_image, float& scale, int& pad_h, int& pad_w){
    /* Letterbox：等比缩放 + 灰边(114)填充到 input_w_ x input_h_ */
    scale = std::min(input_w_ / (float)bgr_image.cols, input_h_ / (float)bgr_image.rows);
    int new_w = cvRound(bgr_image.cols * scale);
    int new_h = cvRound(bgr_image.rows * scale);
    pad_w = (input_w_ - new_w) / 2;
    pad_h = (input_h_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(new_w, new_h));
    cv::copyMakeBorder(resized, preprocess_image,
                       pad_h, input_h_ - new_h - pad_h,
                       pad_w, input_w_ - new_w - pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    /* BGR → RGB（归一化/CHW/打包 rknn_input 不需要：infer 里 UINT8+NHWC 交给驱动） */
    cv::cvtColor(preprocess_image, preprocess_image, cv::COLOR_BGR2RGB);
}


int Yolo::postprocess(const std::vector<rknn_output>& outputs, float scale, int pad_h, int pad_w, std::vector<Detection>& results){
    /* 该模型检测头没拼成 [1,84,8400]，输出是 3 尺度 × (DFL 框 64ch, 类别 80ch, 辅助 1ch)
       自己做 DFL 解码：4 组×bins 加权求和得 (l,t,r,b)（单位 grid），锚点是格心 (x+0.5, y+0.5) */
    for (int o = 0; o + 2 < n_output_; o += 3) {
        const rknn_tensor_attr& battr = output_tensor_attrs_[o];
        const rknn_tensor_attr& cattr = output_tensor_attrs_[o + 1];   // o+2 辅助输出，用不上
        int bc = battr.dims[1], cc = cattr.dims[1];   // 输出 dims = [n, c, h, w]，buf 是 CHW
        int g = battr.dims[2];                       // 格子边长 80/40/20
        int bins = bc / 4, stride = input_w_ / g;
        const float* box = (const float*)outputs[o].buf;
        const float* cls = (const float*)outputs[o + 1].buf;
        auto b_at = [&](int c, int y, int x) { return box[(c * g + y) * g + x]; };
        auto c_at = [&](int c, int y, int x) { return cls[(c * g + y) * g + x]; };

        for (int y = 0; y < g; ++y) {
            for (int x = 0; x < g; ++x) {
                /* 类别概率（模型里已过 sigmoid），取最大 */
                int best = 0;
                float best_score = c_at(0, y, x);
                for (int c = 1; c < cc; ++c) {
                    float v = c_at(c, y, x);
                    if (v > best_score) { best_score = v; best = c; }
                }
                if (best_score < confidence_threshold_) continue;
                /* DFL：每组 bins 个 logits，softmax 后按 0..bins-1 加权求和 */
                float dist[4];
                for (int k = 0; k < 4; ++k) {
                    float m = b_at(k * bins, y, x), sum = 0.f, acc = 0.f;
                    for (int b = 1; b < bins; ++b) m = std::max(m, b_at(k * bins + b, y, x));
                    for (int b = 0; b < bins; ++b) {
                        float e = std::exp(b_at(k * bins + b, y, x) - m);
                        sum += e;
                        acc += e * b;
                    }
                    dist[k] = acc / sum;
                }
                /* (l,t,r,b) → 输入图 xyxy → 去 letterbox 映射回原图；x,y 存左上角（与 iou 约定一致） */
                float cx = (x + 0.5f + (dist[2] - dist[0]) / 2) * stride;
                float cy = (y + 0.5f + (dist[3] - dist[1]) / 2) * stride;
                float w = (dist[0] + dist[2]) * stride / scale, h = (dist[1] + dist[3]) * stride / scale;
                results.push_back({
                    (cx - pad_w) / scale - w / 2,
                    (cy - pad_h) / scale - h / 2,
                    w, h, best, best_score
                });
            }
        }
    }
    nms(results);
    return 0;
}


/* IoU：约定 x,y 为左上角，须与 postprocess 解码一致 */
static float iou(const Detection& a, const Detection& b) {
    cv::Rect2f ra(a.x, a.y, a.w, a.h), rb(b.x, b.y, b.w, b.h);
    cv::Rect2f inter = ra & rb;
    return inter.area() / (ra.area() + rb.area() - inter.area());
}

void Yolo::nms(std::vector<Detection>& dets){
    /* 按 score 降序，逐个保留：与已保留的同类框 IoU 超阈值的丢弃 */
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<Detection> keep;
    for (const auto& det : dets) {
        bool dup = false;
        for (const auto& k : keep)
            if (k.class_id == det.class_id && iou(k, det) > nms_threshold_) { dup = true; break; }
        if (!dup) keep.push_back(det);
    }
    dets = std::move(keep);
}