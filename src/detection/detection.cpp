#include "detection.hpp"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.5;

std::vector<std::string> load_class_names(const std::string& path) 
{
    std::ifstream infile(path);
    std::vector<std::string> class_names;
    std::string line;
    while (std::getline(infile, line)) 
    {
        class_names.push_back(line);
    }
    return class_names;
}

int imageInference(const std::string& inputPath) 
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    #ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    #endif

    Ort::Session session(env, "../bin/yolov5s.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();
    std::vector<const char*> input_names, output_names;
    for (const auto& name : input_names_str) input_names.push_back(name.c_str());
    for (const auto& name : output_names_str) output_names.push_back(name.c_str());

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<std::string> class_names = load_class_names("../resources/coco.names");

    cv::Mat img = cv::imread(inputPath);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3);
    int idx = 0;
    for (int c = 0; c < 3; ++c) 
    {
        for (int y = 0; y < INPUT_HEIGHT; ++y) 
        {
            for (int x = 0; x < INPUT_WIDTH; ++x) 
            {
                input_tensor_values[idx++] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    std::array<int64_t, 4> input_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(), &input_tensor, 1,
                                      output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t num_detections = output_shape[1];
    size_t element_size = output_shape[2];

    std::vector<Detection> detections;

    for (size_t i = 0; i < num_detections; ++i) 
    {
        float obj_conf = output_data[i * element_size + 4];
        if (obj_conf < CONFIDENCE_THRESHOLD) continue;

        float x_center = output_data[i * element_size + 0];
        float y_center = output_data[i * element_size + 1];
        float width    = output_data[i * element_size + 2];
        float height   = output_data[i * element_size + 3];

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
        if (final_conf < CONFIDENCE_THRESHOLD) continue;

        int left = static_cast<int>((x_center - width / 2) * img.cols / INPUT_WIDTH);
        int top  = static_cast<int>((y_center - height / 2) * img.rows / INPUT_HEIGHT);
        int right = static_cast<int>((x_center + width / 2) * img.cols / INPUT_WIDTH);
        int bottom = static_cast<int>((y_center + height / 2) * img.rows / INPUT_HEIGHT);

        cv::Rect box(left, top, right - left, bottom - top);
        detections.push_back({class_id, final_conf, box});
    }

    std::vector<int> nms_indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto& det : detections) 
    {
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }

    float nms_threshold = 0.45;
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, nms_threshold, nms_indices);

    for (int idx : nms_indices) 
    {
        const auto& det = detections[idx];

        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label_ss;
        label_ss << class_names[det.class_id] << " " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label = label_ss.str();

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img, cv::Point(det.box.x, det.box.y - label_size.height - 5),
                            cv::Point(det.box.x + label_size.width, det.box.y),
                            cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
    }

    cv::imshow("Detections", img);
    cv::waitKey(0);
    return 0;
}

int videoInference(const std::string& inputPath) 
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(cuda_options));
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, "../bin/yolov5s.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();
    std::vector<const char*> input_names, output_names;

    for (const auto& name : input_names_str) input_names.push_back(name.c_str());
    for (const auto& name : output_names_str) output_names.push_back(name.c_str());

    std::cout << "Opening video file: " << inputPath << std::endl;
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) 
    {
        std::cerr << "Error: Could not open video file: " << inputPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

        std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3);
        int idx = 0;
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < INPUT_HEIGHT; ++y)
                for (int x = 0; x < INPUT_WIDTH; ++x)
                    input_tensor_values[idx++] = resized.at<cv::Vec3f>(y, x)[c];

        std::array<int64_t, 4> input_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t num_detections = output_shape[1];
        size_t element_size = output_shape[2];

        struct Detection {
            int class_id;
            float confidence;
            cv::Rect box;
        };
        std::vector<Detection> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;

        for (size_t i = 0; i < num_detections; ++i) {
            float x_center = output_data[i * element_size + 0];
            float y_center = output_data[i * element_size + 1];
            float width    = output_data[i * element_size + 2];
            float height   = output_data[i * element_size + 3];
            float obj_conf = output_data[i * element_size + 4];

            if (obj_conf < CONFIDENCE_THRESHOLD)
                continue;

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
            if (final_conf < CONFIDENCE_THRESHOLD)
                continue;

            int left   = static_cast<int>((x_center - width / 2) * frame.cols / INPUT_WIDTH);
            int top    = static_cast<int>((y_center - height / 2) * frame.rows / INPUT_HEIGHT);
            int right  = static_cast<int>((x_center + width / 2) * frame.cols / INPUT_WIDTH);
            int bottom = static_cast<int>((y_center + height / 2) * frame.rows / INPUT_HEIGHT);

            cv::Rect box(left, top, right - left, bottom - top);
            detections.push_back({class_id, final_conf, box});
            boxes.push_back(box);
            scores.push_back(final_conf);
        }

        std::vector<int> nms_indices;
        float nms_threshold = 0.45;
        cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, nms_threshold, nms_indices);

        for (int idx : nms_indices) {
            const auto& det = detections[idx];
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, std::to_string(det.class_id), cv::Point(det.box.x, det.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }

        cv::imshow("YOLOv5 - Video", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

int inference(const std::string& inputPath) 
{
    if (inputPath.size() >= 4) 
    {
        std::string ext = inputPath.substr(inputPath.size() - 4);
        if (ext == ".jpg" || ext == ".JPG" || ext == ".png" || ext == ".PNG") 
        {
            return imageInference(inputPath);
        } 
        else if (ext == ".mp4" || ext == ".MP4" || ext == ".avi" || ext == ".AVI") 
        {
            return videoInference(inputPath);
        }
    }
    return -1;
}