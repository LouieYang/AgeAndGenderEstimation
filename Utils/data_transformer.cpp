#include "data_transformer.hpp"

Eigen::MatrixXf OpenCV2Eigen(const cv::Mat& image)
{
    cv::Mat imageGray;
    if (image.channels() != 1)
    {
        cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    }
    else
    {
        imageGray = cv::Mat(image);
    }
    imageGray.convertTo(imageGray, CV_32FC1);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> imageMatrix(imageGray.ptr<float>(), imageGray.rows, imageGray.cols);
    
    return std::move(imageMatrix);
}