#ifndef gender_estimation_hpp
#define gender_estimation_hpp

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "tool_box.hpp"

/**
 *  @Brief: This module implements the gender estimation with caffe
 *
 *  @Paper Reference: Age and Gender Classification using Convolutional 
 *         Neural Networks (CVPR 2015)
 *
 *  The reason why merge this function with age estimation are presented
 *  in age_estimation.hpp
 *
 */

enum class Gender {Man, Woman, Error};

/**
 *  Using static to eliminate the linking error
 */
static std::map<Gender, std::string> gender_listss
{
    {Gender::Man, "Man"},
    {Gender::Woman, "Woman"},
    {Gender::Error, "Error"}
};

std::vector<Gender> gender_estimation(const cv::Mat&, const std::vector<BoundingBox>&);

#endif
