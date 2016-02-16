#ifndef face_detection_hpp
#define face_detection_hpp

#include <string>
#include <vector>
#include <fstream>

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "data_transformer.hpp"
#include "bounding_box.hpp"

/**
 *  @Const Value given by imageNet mean image
 */
constexpr float red_channel_mean   = 104.00698793;
constexpr float green_channel_mean = 116.66876762;
constexpr float blue_channel_mean  = 122.67891434;

/**
 *  @Brief: face detection function using caffe network
 *
 *  @Paper Reference: Implementation: Multi-view Face Detection using deep
 *                                    convolutional neural networks
 *
 *  @param cv::Mat&                  The original image passed
 *  @param std::vector<BoundingBox>& The face regions in the image
 */
void face_detection(const cv::Mat&, std::vector<BoundingBox>&);

/**
 *  @Brief: Change the input dim in the prototxt files
 *
 *  @param int: The row dim
 *  @param int: The col dim
 */
void modify_prototxt(const int, const int);

/**
 *  @Brief: Change the caffe result into prototype bounding box
 *
 *  @param Eigen::MatrixXf&          Result from caffe output layer
 *  @param float                     Threshold that judge the right area
 *  @param std::vector<BoundingBox>& The returning bounding box
 */
void add_to_boundingbox(const Eigen::MatrixXf&, const float,
                        std::vector<BoundingBox>&);


#endif /* face_detection_hpp */
