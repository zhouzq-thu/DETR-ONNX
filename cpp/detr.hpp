//
//  detr.hpp
//  detr_onnx
//
//  Created by Zhi-Qiang Zhou on 2025/11/10.
//  Copyright © 2025 Zhi-Qiang Zhou. All rights reserved.
//

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

// Helper: parse class names from ONNX metadata string like "{'0': 'person',
// '1': 'bicycle', ...}"
std::map<int, std::string> parseClassNames(const std::string &namesStr) {
  std::map<int, std::string> result;
  if (namesStr.empty())
    return result;

  // Very simple parser for Python dict string (assumes format: "{'0': 'name',
  // ...}")
  std::istringstream iss(namesStr);
  std::string token;
  while (std::getline(iss, token, ',')) {
    size_t colon = token.find(':');
    if (colon == std::string::npos)
      continue;

    std::string idPart = token.substr(0, colon);
    std::string namePart = token.substr(colon + 1);

    // Extract number between quotes or braces
    auto clean = [](std::string s) {
      s.erase(std::remove(s.begin(), s.end(), '\''), s.end());
      s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
      s.erase(std::remove(s.begin(), s.end(), '{'), s.end());
      s.erase(std::remove(s.begin(), s.end(), '}'), s.end());
      return s;
    };

    idPart = clean(idPart);
    namePart = clean(namePart);

    try {
      int id = std::stoi(idPart);
      result[id] = namePart;
    } catch (...) {
    }
  }
  return result;
}

// Softmax for 2D matrix (rows = queries, cols = classes)
std::vector<std::vector<float>>
softmax2d(const std::vector<std::vector<float>> &logits) {
  std::vector<std::vector<float>> probas(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const auto &row = logits[i];
    float max_val = *std::max_element(row.begin(), row.end());
    float sum = 0.0f;
    std::vector<float> exps(row.size());
    for (size_t j = 0; j < row.size(); ++j) {
      exps[j] = std::exp(row[j] - max_val);
      sum += exps[j];
    }
    probas[i].resize(row.size());
    for (size_t j = 0; j < row.size(); ++j) {
      probas[i][j] = exps[j] / sum;
    }
  }
  return probas;
}

// Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
std::vector<float> cxcywh_to_xyxy(const std::vector<float> &box) {
  float cx = box[0], cy = box[1], w = box[2], h = box[3];
  return {cx - 0.5f * w, cy - 0.5f * h, cx + 0.5f * w, cy + 0.5f * h};
}

class DetrONNX {
public:
  explicit DetrONNX(const std::string &modelPath) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session =
        std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_count = session->GetInputCount();
    auto output_count = session->GetOutputCount();
    for (size_t i = 0; i < input_count; ++i) {
      inputNames.emplace_back(
          strdup(session->GetInputNameAllocated(i, allocator).get()));
    }
    for (size_t i = 0; i < output_count; ++i) {
      outputNames.emplace_back(
          strdup(session->GetOutputNameAllocated(i, allocator).get()));
    }

    // Get input shape
    auto inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();
    inputShape_ = inputShape;

    // Load metadata
    auto modelMetadata = session->GetModelMetadata();
    auto keys = modelMetadata.GetCustomMetadataMapKeysAllocated(allocator);
    for (const auto &k : keys) {
      std::string key = k.get();
      auto ptr = modelMetadata.LookupCustomMetadataMapAllocated(key.c_str(),
                                                                allocator);
      if (ptr != nullptr)
        this->metadata[key] = ptr.get();
    }

    // Parse metadata
    stride = std::stoi(metadata.count("stride") ? metadata["stride"] : "-1");
    useFocalLoss =
        (metadata.count("use_focal_loss") && metadata["use_focal_loss"] == "1");
    classNames =
        parseClassNames(metadata.count("names") ? metadata["names"] : "");
  }

  ~DetrONNX() {
    for (auto &name : inputNames)
      free((void *)name);
    for (auto &name : outputNames)
      free((void *)name);
  }

  struct Detection {
    cv::Rect2f box; // x1, y1, x2, y2
    int classId;
    float confidence;
  };

  std::vector<Detection> detect(const cv::Mat &image,
                                float probThreshold = 0.7f) {
    int targetH, targetW;

    // Determine input size
    if (inputShape_[2] < 0 || inputShape_[3] < 0) {
      // Dynamic shape: use stride-based rounding or default
      if (stride > 0) {
        targetH = (image.rows + stride - 1) / stride * stride;
        targetW = (image.cols + stride - 1) / stride * stride;
      } else {
        targetH = 800;
        targetW = 800;
      }
    } else {
      targetH = static_cast<int>(inputShape_[2]);
      targetW = static_cast<int>(inputShape_[3]);
    }

    // Preprocess
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetW, targetH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    // HWC to CHW
    size_t inputTensorSize = 3 * targetH * targetW;
    std::vector<float> inputTensorValues(inputTensorSize);
    std::vector<cv::Mat> channels = {
        cv::Mat(targetH, targetW, CV_32FC1, inputTensorValues.data()),
        cv::Mat(targetH, targetW, CV_32FC1, inputTensorValues.data() + targetH * targetW),
        cv::Mat(targetH, targetW, CV_32FC1, inputTensorValues.data() + 2 * targetH * targetW)
    };
    cv::split(resized, channels);

    // Create input tensor
    std::vector<int64_t> inputDims = {1, 3, targetH, targetW};
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputDims.data(), inputDims.size());

    // Run inference
    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                     inputNames.size(), outputNames.data(), outputNames.size());

    // Assume [scores, boxes] — common in DETR
    float *scoresData = outputTensors[0].GetTensorMutableData<float>();
    float *boxesData = outputTensors[1].GetTensorMutableData<float>();

    auto scoresShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto boxesShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int numQueries = static_cast<int>(scoresShape[1]);
    int numClasses = static_cast<int>(scoresShape[2]); // includes "no-object"

    // Extract scores (exclude last class = no-object)
    std::vector<std::vector<float>> logits(numQueries,
                                           std::vector<float>(numClasses - 1));
    for (int i = 0; i < numQueries; ++i) {
      for (int c = 0; c < numClasses - 1; ++c) {
        logits[i][c] = scoresData[i * numClasses + c];
      }
    }

    auto probas = softmax2d(logits);

    std::vector<Detection> detections;
    for (int i = 0; i < numQueries; ++i) {
      float maxProb = *std::max_element(probas[i].begin(), probas[i].end());
      if (maxProb > probThreshold) {
        int classId = std::max_element(probas[i].begin(), probas[i].end()) -
                      probas[i].begin();
        std::vector<float> rawBox(4);
        for (int j = 0; j < 4; ++j) {
          rawBox[j] = boxesData[i * 4 + j];
        }
        auto xyxy = cxcywh_to_xyxy(rawBox);
        detections.push_back(
            {cv::Rect2f(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]),
             classId, maxProb});
      }
    }

    for (auto &det : detections) {
      det.box.x *= image.cols;
      det.box.y *= image.rows;
      det.box.width *= image.cols;
      det.box.height *= image.rows;
    }

    return detections;
  }

  void plotResult(cv::Mat &image, const std::vector<Detection> &detections) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (const auto &det : detections) {
      cv::Scalar color(dis(gen), dis(gen), dis(gen));
      cv::rectangle(image, det.box, color, 2);

      std::string label;
      if (classNames.count(det.classId)) {
        label = classNames[det.classId];
      } else {
        label = std::to_string(det.classId);
      }
      std::string txt = label + ": " +
                        std::to_string(static_cast<int>(det.confidence * 100)) +
                        "%";
      int baseline = 0;
      auto textSize =
          cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
      cv::Point tl(det.box.x, std::max(15, static_cast<int>(det.box.y - 5)));
      cv::Point br(tl.x + textSize.width, tl.y + textSize.height + baseline);

      cv::rectangle(image, tl, br, color, -1);
      cv::putText(image, txt, cv::Point(tl.x, tl.y + textSize.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(255), 2);
    }
  }

private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DETR"};
  std::unique_ptr<Ort::Session> session;
  std::vector<const char *> inputNames, outputNames;
  std::vector<int64_t> inputShape_;
  std::map<std::string, std::string> metadata;
  int stride = -1;
  bool useFocalLoss = false;
  std::map<int, std::string> classNames;
};
