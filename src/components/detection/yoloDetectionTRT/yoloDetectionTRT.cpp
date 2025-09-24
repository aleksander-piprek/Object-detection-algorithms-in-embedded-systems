#include "yoloDetectionTRT.hpp"
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <sstream>
#include <NvInferRuntime.h>

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define CONFIDENCE_THRESHOLD 0.4f
#define NMS_THRESHOLD 0.45f

using namespace nvinfer1;

namespace {
// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
} gLogger;

// Utility to check CUDA errors
#define CHECK_CUDA(status) do { \
    if (status != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(status))); \
    } \
} while (0)

// Letterbox preprocessing
void letterbox(const cv::Mat& input, cv::Mat& output, int target_w, int target_h) {
    float scale = std::min((float)target_w / input.cols, (float)target_h / input.rows);
    int new_w = static_cast<int>(input.cols * scale);
    int new_h = static_cast<int>(input.rows * scale);
    
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    output = cv::Mat(target_h, target_w, input.type(), cv::Scalar(114, 114, 114)); // Gray padding
    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;
    resized.copyTo(output(cv::Rect(x_offset, y_offset, new_w, new_h)));
}
}

YoloDetectionTRT::YoloDetectionTRT(const std::string& engine_path, const std::string& class_names_path) {
    // Read TensorRT engine
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    size_t size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);
    file.close();

    // Initialize TensorRT
    runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // Get binding indices
    inputIndex = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");
    if (inputIndex == -1 || outputIndex == -1) {
        throw std::runtime_error("Invalid binding indices: input=" + std::to_string(inputIndex) + ", output=" + std::to_string(outputIndex));
    }

    // Get input dimensions
    auto input_dims = engine->getBindingDimensions(inputIndex);
    if (input_dims.nbDims != 4) {
        throw std::runtime_error("Expected 4D input tensor, got " + std::to_string(input_dims.nbDims));
    }
    inputH = input_dims.d[2];
    inputW = input_dims.d[3];

    // Allocate buffers
    CHECK_CUDA(cudaMalloc(&buffers[inputIndex], 3 * inputH * inputW * sizeof(float)));
    auto output_dims = engine->getBindingDimensions(outputIndex);
    output_size = output_dims.d[1] * output_dims.d[2]; // e.g., 25200 * 85 for YOLOv5s
    CHECK_CUDA(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

    // Create CUDA stream
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Load class names
    class_names = loadClassNames(class_names_path);
}

YoloDetectionTRT::~YoloDetectionTRT() {
    if (stream) {
        cudaStreamDestroy(stream);
    }
    for (int i = 0; i < 2; ++i) {
        if (buffers[i]) {
            cudaFree(buffers[i]);
        }
    }
}

std::vector<std::string> YoloDetectionTRT::loadClassNames(const std::string& path) {
    std::ifstream infile(path);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open class names file: " + path);
    }
    std::vector<std::string> names;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    infile.close();
    return names;
}

std::vector<DetectionResult> YoloDetectionTRT::inference(cv::Mat& frame) {
    // Letterbox preprocessing
    cv::Mat preprocessed;
    letterbox(frame, preprocessed, INPUT_WIDTH, INPUT_HEIGHT);

    // Convert to float and normalize
    cv::Mat blob;
    preprocessed.convertTo(blob, CV_32F, 1.0 / 255.0);
    if (blob.channels() != 3) {
        throw std::runtime_error("Expected 3-channel input image");
    }

    // Copy to GPU
    CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], blob.ptr<float>(), 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Run inference
    context->enqueueV2(buffers, stream, nullptr);

    // Copy output to host
    std::vector<float> output(output_size);
    CHECK_CUDA(cudaMemcpyAsync(output.data(), buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Parse YOLOv5 output (e.g., [1, 25200, 85] for 80 classes)
    std::vector<DetectionResult> detections, finalDetections;
    int num_detections = output_size / (5 + class_names.size());
    float scale_x = (float)frame.cols / INPUT_WIDTH;
    float scale_y = (float)frame.rows / INPUT_HEIGHT;

    for (int i = 0; i < num_detections; ++i) {
        float obj_conf = output[i * (5 + class_names.size()) + 4];
        if (obj_conf < CONFIDENCE_THRESHOLD) continue;

        float x = output[i * (5 + class_names.size()) + 0];
        float y = output[i * (5 + class_names.size()) + 1];
        float w = output[i * (5 + class_names.size()) + 2];
        float h = output[i * (5 + class_names.size()) + 3];

        int class_id = 0;
        float max_score = 0.0f;
        for (int j = 5; j < 5 + class_names.size(); ++j) {
            float score = output[i * (5 + class_names.size()) + j];
            if (score > max_score) {
                max_score = score;
                class_id = j - 5;
            }
        }

        float conf = obj_conf * max_score;
        if (conf < CONFIDENCE_THRESHOLD) continue;

        // Adjust for letterbox scaling
        int left = static_cast<int>((x - w / 2) * scale_x);
        int top = static_cast<int>((y - h / 2) * scale_y);
        int width = static_cast<int>(w * scale_x);
        int height = static_cast<int>(h * scale_y);

        detections.push_back({class_id, conf, cv::Rect(left, top, width, height)});
    }

    // Non-Max Suppression
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    for (const auto& d : detections) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int idx : indices) {
        finalDetections.push_back(detections[idx]);
    }

    return finalDetections;
}

void YoloDetectionTRT::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) {
    for (const auto& d : detections) {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
        std::ostringstream ss;
        ss << class_names[d.class_id] << " " << std::fixed << std::setprecision(2) << d.confidence;
        std::string label = ss.str();

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(d.box.x, d.box.y - label_size.height - 5),
                      cv::Point(d.box.x + label_size.width, d.box.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(d.box.x, d.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}