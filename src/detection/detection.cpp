#include "detection.hpp"

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.3;

Detection::Detection(const std::string& model_path, const std::string& class_names_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolov5"), session_options()
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_names_str = session->GetInputNames();
    output_names_str = session->GetOutputNames();

    input_names.clear();
    output_names.clear();

    for (const auto& name : input_names_str) {
        input_names.push_back(name.c_str());
    }
    for (const auto& name : output_names_str) {
        output_names.push_back(name.c_str());
    }

    class_names = loadClassNames(class_names_path);
}

std::vector<std::string> Detection::loadClassNames(const std::string& path)
{
    std::ifstream infile(path);
    std::vector<std::string> class_names;
    std::string line;
    while (std::getline(infile, line)) 
        class_names.push_back(line);
    return class_names;
}

std::vector<DetectionResult> Detection::inference(const cv::Mat& frame)
{
    cv::Mat resized, rgb_float;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(rgb_float, CV_32FC3, 1.0f / 255.0f);

    std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3);
    int idx = 0;
    for (int c = 0; c < 3; ++c) 
    {
        for (int y = 0; y < INPUT_HEIGHT; ++y) 
        {
            for (int x = 0; x < INPUT_WIDTH; ++x) 
            {
                input_tensor_values[idx++] = rgb_float.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::array<int64_t, 4> input_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_dims.data(), input_dims.size());

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       input_names.data(), &input_tensor, 1,
                                       output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t num_detections = output_shape[1];
    size_t element_size = output_shape[2];

    std::vector<DetectionResult> detections;

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
        if (final_conf < CONFIDENCE_THRESHOLD) continue;

        int left = static_cast<int>((x_center - width / 2) * frame.cols / INPUT_WIDTH);
        int top  = static_cast<int>((y_center - height / 2) * frame.rows / INPUT_HEIGHT);
        int right = static_cast<int>((x_center + width / 2) * frame.cols / INPUT_WIDTH);
        int bottom = static_cast<int>((y_center + height / 2) * frame.rows / INPUT_HEIGHT);

        cv::Rect box(left, top, right - left, bottom - top);
        detections.push_back({class_id, final_conf, box});
    }

    return detections;
}

void Detection::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections)
{
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
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label_ss;
        label_ss << class_names[det.class_id] << " " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label = label_ss.str();

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(det.box.x, det.box.y - label_size.height - 5),
                      cv::Point(det.box.x + label_size.width, det.box.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
    }
}
