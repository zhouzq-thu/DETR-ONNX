//
//  main.cpp
//  detr_onnx
//
//  Created by Zhi-Qiang Zhou on 2025/11/10.
//  Copyright © 2025 Zhi-Qiang Zhou. All rights reserved.
//

#include "detr.hpp"

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    try {
        std::string modelPath = argv[1];
        std::string imagePath = argv[2];

        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Cannot read image: " << imagePath << std::endl;
            return -1;
        }

        DetrONNX detr(modelPath);
        auto detections = detr.detect(img, 0.7f);

        std::cout << "Found " << detections.size() << " objects." << std::endl;

        cv::Mat out = img.clone();
        detr.plotResult(out, detections);

        cv::imshow("DETR (ONNX Runtime)", out);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
