#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Path to the image file
    std::string imagePath = "/Users/jongbeomkim/Downloads/KakaoTalk_Photo_2024-08-15-16-05-05.jpeg";

    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR); // Load as a color image

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Loaded Image", image);

    // Wait for a key press indefinitely
    cv::waitKey(0);

    return 0;
}
