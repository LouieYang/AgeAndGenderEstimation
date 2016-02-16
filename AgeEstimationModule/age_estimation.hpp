#ifndef age_estimation_hpp
#define age_estimation_hpp

#include <vector>

#include <opencv2/opencv.hpp>

#include "tool_box.hpp"

/**
 *  @Brief: The age estimation module using trained caffe
 *
 *  @Paper Reference: Age and Gender Classification using Convolutional
 *         Neural Networks (CVPR 2015)
 *
 *  This function can also be merged with gender estimation for efficient.
 *  However, it is better to seperate them since the user may use just
 *  single one, in other word, either gender estimation or age estimation
 *
 */
enum class Age
{
    R0_2, R4_6, R8_13, R15_20, R25_32, R38_43, R48_53, R60_, Error
};

/**
 *  Using static to solve the linking error
 */
static std::map<Age, std::string> age_list
{
    {Age::R0_2, "0~2 years old"},
    {Age::R4_6, "4~6 years old"},
    {Age::R8_13, "8~13 years old"},
    {Age::R15_20, "15~20 years old"},
    {Age::R25_32, "25~32 years old"},
    {Age::R38_43, "38~43 years old"},
    {Age::R48_53, "48~53 years old"},
    {Age::R60_, "60+ years old"},
    {Age::Error, "Error"}
};


std::vector<Age> age_estimation(const cv::Mat& Image, const std::vector<BoundingBox>& face_rois);

#endif /* age_estimation_hpp */
