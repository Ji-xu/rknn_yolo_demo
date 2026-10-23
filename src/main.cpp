#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "inner/yolo.h"
#include "inner/coco_classes.h"

#define MODEL_PATH "../model/yolov8.rknn"   /* 相对 bin/ 运行目录 */

static volatile sig_atomic_t g_stop = 0;
static void on_stop(int) { g_stop = 1; }

/* 整帧写入 ffmpeg 管道；管道满则丢帧（rawvideo 必须整帧写完，写一半会错位） */
static bool push_frame(int fd, const cv::Mat& frame)
{
    pollfd pfd = {fd, POLLOUT, 0};
    if (poll(&pfd, 1, 0) <= 0) return false;
    const char* p = (const char*)frame.data;
    size_t size = frame.total() * frame.elemSize();
    for (size_t done = 0; done < size; ) {
        ssize_t n = write(fd, p + done, size - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            static bool warned = false;
            if (!warned) { printf("Pipe write failed: %s\n", strerror(errno)); warned = true; }
            return false;
        }
        done += n;
    }
    return true;
}

int main()
{
    setenv("OPENCV_LOG_LEVEL", "ERROR", 0);   /* 压掉 GStreamer 后端的无害 WARN */
    /* 自己拉一个 MediaMTX 当 RTSP 服务器；绑不上 8554（已在跑）它会自行退出，不打架 */
    if (fork() == 0) {
        execlp("mediamtx", "mediamtx", "../mediamtx.yml", (char*)NULL);
        perror("mediamtx");
        _exit(127);
    }
    /* shortcut: 靠模型加载耗时兜住 mediamtx 监听竞态, 若改成秒开需等 8554 就绪再推流 */
    /* 创建 yolo 对象并初始化 */
    Yolo yolo(MODEL_PATH);
    if (yolo.init() != 0) {
        return -1;
    }
    /* 打开 USB 相机 */
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Failed to open camera 0\n");
        return -1;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);     // 驱动缓冲压到 1 帧，尽量取最新
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    /* 启动 ffmpeg：H.264 编码后推给本机 MediaMTX（这版 ffmpeg 的 rtsp muxer 只会推流，不会 listen）
       VLC 拉 rtsp://<板子IP>:8554/live；换 ffmpeg-rockchip 后把 libx264 换成 h264_rkmpp 即硬编 */
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -hide_banner -loglevel error -nostats -f rawvideo -pix_fmt bgr24 -s %dx%d -r 25 -i - "
             "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 25 -an "
             "-f rtsp rtsp://127.0.0.1:8554/live", w, h);
    FILE* ff = popen(cmd, "w");
    if (!ff) {
        printf("Failed to start ffmpeg\n");
        return -1;
    }
    int out_fd = fileno(ff);
    signal(SIGINT, on_stop);
    signal(SIGPIPE, SIG_IGN);    // ffmpeg 退出后 write 得到 EPIPE，不要打死进程

    cv::Mat frame;
    std::vector<Detection> detections;
    int frames = 0, dropped = 0;
    double t0 = (double)cv::getTickCount();
    while (!g_stop && cap.read(frame)) {
        /* 推理 */
        if (yolo.infer(frame, detections) != 0) break;
        /* 画框 */
        for (const auto& d : detections) {
            cv::Rect box((int)d.x, (int)d.y, (int)d.w, (int)d.h);
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", coco_cls_name(d.class_id), d.score);
            cv::putText(frame, label, {box.x, box.y - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        /* 推流 */
        if (!push_frame(out_fd, frame)) ++dropped;
        if (++frames % 30 == 0) {
            double fps = 30.0 * cv::getTickFrequency() / ((double)cv::getTickCount() - t0);
            printf("FPS: %.1f, detections: %d, dropped: %d\n", fps, (int)detections.size(), dropped);
            t0 = (double)cv::getTickCount();
        }
    }
    pclose(ff);
    return 0;
}
