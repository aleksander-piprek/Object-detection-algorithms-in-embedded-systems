#include "yoloDetectionTRT.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime_api.h>

YoloDetectionTRT::YoloDetectionTRT(const std::string& engine_path, 
                                   const std::string& class_names_path)
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(nullptr)
{
    std::cout << "Created YoloDetectionTRT" << std::endl << std::flush;

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    cudaStreamCreate(&stream);

    int32_t num_bindings = engine->getNbIOTensors();
    buffers.resize(num_bindings);
    binding_names.resize(num_bindings);
    is_input.resize(num_bindings);
    binding_dims.resize(num_bindings);

    for (int32_t i = 0; i < num_bindings; ++i) {
        binding_names[i] = engine->getIOTensorName(i);
        is_input[i] = engine->getTensorIOMode(binding_names[i].c_str()) == nvinfer1::TensorIOMode::kINPUT;
        binding_dims[i] = engine->getTensorShape(binding_names[i].c_str());
        size_t size = 1;
        for (int j = 0; j < binding_dims[i].nbDims; ++j) {
            size *= binding_dims[i].d[j];
        }
        size *= sizeof(float);
        cudaMalloc(&buffers[i], size);
    }

    class_names = loadClassNames(class_names_path);
}

YoloDetectionTRT::YoloDetectionTRT(const std::string& engine_path, 
                                   const std::string& class_names_path, 
                                   const float& confThreshold, 
                                   const float& nmsThreshold)
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(nullptr),
      CONFIDENCE_THRESHOLD(confThreshold), NMS_THRESHOLD(nmsThreshold)
{
    std::cout << "Created YoloDetectionTRT" << std::endl << std::flush;

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    cudaStreamCreate(&stream);

    int32_t num_bindings = engine->getNbIOTensors();
    buffers.resize(num_bindings);
    binding_names.resize(num_bindings);
    is_input.resize(num_bindings);
    binding_dims.resize(num_bindings);

    for (int32_t i = 0; i < num_bindings; ++i) {
        binding_names[i] = engine->getIOTensorName(i);
        is_input[i] = engine->getTensorIOMode(binding_names[i].c_str()) == nvinfer1::TensorIOMode::kINPUT;
        binding_dims[i] = engine->getTensorShape(binding_names[i].c_str());
        size_t size = 1;
        for (int j = 0; j < binding_dims[i].nbDims; ++j) {
            size *= binding_dims[i].d[j];
        }
        size *= sizeof(float);
        cudaMalloc(&buffers[i], size);
    }

    class_names = loadClassNames(class_names_path);
}

YoloDetectionTRT::~YoloDetectionTRT()
{
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

std::vector<std::string> YoloDetectionTRT::loadClassNames(const std::string& path)
{
    std::ifstream infile(path);
    std::vector<std::string> class_names;
    std::string line;
    while (std::getline(infile, line)) {
        class_names.push_back(line);
    }
    return class_names;
}

std::vector<DetectionResult> YoloDetectionTRT::inference(cv::Mat& frame)
{
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        1.0f / 255.0f,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(0, 0, 0),
        true,
        false,
        CV_32F
    );

    if (!blob.isContinuous()) {
        blob = blob.clone();
    }

    size_t input_size = blob.total() * sizeof(float);
    int input_idx = -1;
    for (size_t i = 0; i < is_input.size(); ++i) {
        if (is_input[i]) {
            input_idx = i;
            break;
        }
    }
    if (input_idx == -1) {
        throw std::runtime_error("No input binding found");
    }
    cudaMemcpyAsync(buffers[input_idx], blob.data, input_size, cudaMemcpyHostToDevice, stream);

    for (size_t i = 0; i < binding_names.size(); ++i) {
        context->setTensorAddress(binding_names[i].c_str(), buffers[i]);
    }
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    int output_idx = -1;
    for (size_t i = 0; i < is_input.size(); ++i) {
        if (!is_input[i]) {
            output_idx = i;
            break;
        }
    }
    if (output_idx == -1) {
        throw std::runtime_error("No output binding found");
    }
    std::vector<float> output_data;
    size_t output_size = 1;
    for (int i = 0; i < binding_dims[output_idx].nbDims; ++i) {
        output_size *= binding_dims[output_idx].d[i];
    }
    output_data.resize(output_size);
    cudaMemcpyAsync(output_data.data(), buffers[output_idx], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    size_t num_detections = binding_dims[output_idx].d[1];
    size_t element_size = binding_dims[output_idx].d[2];

    std::vector<DetectionResult> detections;
    std::vector<DetectionResult> finalDetections;

    for (size_t i = 0; i < num_detections; ++i) {
        float obj_conf = output_data[i * element_size + 4];
        if (obj_conf < CONFIDENCE_THRESHOLD) {
            continue;
        }

        float x_center = output_data[i * element_size + 0];
        float y_center = output_data[i * element_size + 1];
        float width = output_data[i * element_size + 2];
        float height = output_data[i * element_size + 3];

        int class_id = 0;
        float max_score = 0.0f;
        for (int j = 5; j < element_size; ++j) {
            float score = output_data[i * element_size + j];
            if (score > max_score) {
                max_score = score;
                class_id = j - 5;
            }
        }

        float final_conf = obj_conf * max_score;
        if (final_conf < CONFIDENCE_THRESHOLD) {
            continue;
        }

        int left = static_cast<int>((x_center - width / 2) * frame.cols / INPUT_WIDTH);
        int top = static_cast<int>((y_center - height / 2) * frame.rows / INPUT_HEIGHT);
        int right = static_cast<int>((x_center + width / 2) * frame.cols / INPUT_WIDTH);
        int bottom = static_cast<int>((y_center + height / 2) * frame.rows / INPUT_HEIGHT);

        cv::Rect box(left, top, right - left, bottom - top);
        detections.push_back({class_id, final_conf, box});
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto& det : detections) {
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    for (int idx : nms_indices) {
        finalDetections.push_back(detections[idx]);
    }

    return finalDetections;
}

void YoloDetectionTRT::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections)
{
    for (const auto& detection : detections) {
        cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label_ss;
        label_ss << class_names[detection.class_id] << " " << std::fixed << std::setprecision(2) << detection.confidence;
        std::string label = label_ss.str();

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(detection.box.x, detection.box.y - label_size.height - 5),
                      cv::Point(detection.box.x + label_size.width, detection.box.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(detection.box.x, detection.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}