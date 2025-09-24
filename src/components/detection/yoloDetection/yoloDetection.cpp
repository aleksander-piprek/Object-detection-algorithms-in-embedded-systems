#include "yoloDetection.hpp"

#include "thread"
#include <iostream>
#include <fstream>

YoloDetection::YoloDetection(const std::string& model_path, 
                             const std::string& class_names_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolov5"), session_options()
{
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_names_str = session->GetInputNames();
    output_names_str = session->GetOutputNames();

    input_names.clear();
    output_names.clear();

    for (const auto& name : input_names_str) 
    {
        input_names.push_back(name.c_str());
    }
    
    for (const auto& name : output_names_str) 
    {
        output_names.push_back(name.c_str());
    }

    class_names = loadClassNames(class_names_path);
}

YoloDetection::YoloDetection(const std::string& model_path, 
                             const std::string& class_names_path, 
                             const float& confThreshold, 
                             const float& nmsThreshold)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolov5"), session_options(), CONFIDENCE_THRESHOLD(confThreshold), NMS_THRESHOLD(nmsThreshold)
{
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_names_str = session->GetInputNames();
    output_names_str = session->GetOutputNames();

    input_names.clear();
    output_names.clear();

    for (const auto& name : input_names_str) 
    {
        input_names.push_back(name.c_str());
    }
    
    for (const auto& name : output_names_str) 
    {
        output_names.push_back(name.c_str());
    }

    class_names = loadClassNames(class_names_path);
}

std::vector<std::string> YoloDetection::loadClassNames(const std::string& path)
{
    std::ifstream infile(path);
    std::vector<std::string> class_names;
    std::string line;
    while (std::getline(infile, line)) 
        class_names.push_back(line);
    return class_names;
}

std::vector<DetectionResult> YoloDetection::inference(cv::Mat& frame)
{
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,                 
        1.0f / 255.0f,        
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(0,0,0),    
        true,     
        false,    
        CV_32F                
    );

    if (!blob.isContinuous()) 
        blob = blob.clone();

    int64_t batch = blob.size[0];
    int64_t channels = blob.size[1];
    int64_t height = blob.size[2];
    int64_t width = blob.size[3];

    std::array<int64_t,4> input_dims = { batch, channels, height, width };

    size_t input_tensor_size = static_cast<size_t>(batch * channels * height * width);

    float* blobData = reinterpret_cast<float*>(blob.data);
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(input_tensor_size);
    std::memcpy(input_tensor_values.data(), blobData, input_tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims.data(),
        input_dims.size()
    );

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                    input_names.data(), &input_tensor, 1,
                                    output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t num_detections = output_shape[1];
    size_t element_size = output_shape[2];

    std::vector<DetectionResult> detections;
    std::vector<DetectionResult> finalDetections;    
    
    for (size_t i = 0; i < num_detections; ++i) 
    {
        float obj_conf = output_data[i * element_size + 4];
        if (obj_conf < CONFIDENCE_THRESHOLD) 
            continue;

        float x_center = output_data[i * element_size + 0];
        float y_center = output_data[i * element_size + 1];
        float width    = output_data[i * element_size + 2];
        float height   = output_data[i * element_size + 3];

        int class_id = 0;
        float max_score = 0.0f;
        for (int j = 5; j < element_size; ++j) 
        {
            float score = output_data[i * element_size + j];
            if (score > max_score) 
            {
                max_score = score;
                class_id = j - 5;
            }
        }

        float final_conf = obj_conf * max_score;
        if (final_conf < CONFIDENCE_THRESHOLD) 
            continue;

        int left = int((x_center - width/2) * frame.cols / INPUT_WIDTH);
        int top  = int((y_center - height / 2) * frame.rows / INPUT_HEIGHT);
        int right = int((x_center + width / 2) * frame.cols / INPUT_WIDTH);
        int bottom = int((y_center + height / 2) * frame.rows / INPUT_HEIGHT);

        cv::Rect box(left, top, right - left, bottom - top);
        detections.push_back({class_id, 
                              final_conf, 
                              box});
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto& det : detections)
    {        
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    for (int idx : nms_indices)
    {
        finalDetections.push_back(detections[idx]);            
    }

    return finalDetections;
}

void YoloDetection::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections)
{
    for (const auto& detection : detections) 
    {
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
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
    }
}
