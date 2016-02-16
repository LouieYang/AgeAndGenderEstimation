#ifndef data_transformer_hpp
#define data_transformer_hpp

#ifndef CPU_ONLY
#define CPU_ONLY
#endif

#include "opencv2/opencv.hpp"

#include <eigen3/Eigen/Eigen>

#include <caffe/caffe.hpp>

/**
 *  @Warning: The project root must be changed
 */
const std::string project_root("/Users/liuyang/Desktop/AgeAndGenderEstimation/Source/");

/**
 *  @Brief: The data transform from OpenCV to Eigen without copy
 *
 *  @param image: OpenCV Mat data
 *
 *  @return Eigen::Matrix data
 */
Eigen::MatrixXf OpenCV2Eigen(const cv::Mat& image);

/**
 *  @Brief: Translate the Eigen Mat to caffe input blob using copy function
 *
 *  @param imgs: The data in the Eigen Mat
 *  @param net:  The destination of data transferring
 *
 *  @Warning: Template function must be defined in the .hpp file to avoid
 *            linking error
 */
template <typename Dtype>
void Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs, std::shared_ptr<caffe::Net<Dtype>> net)
{
    caffe::Blob<Dtype>* input_layer = net->input_blobs()[0];
    Dtype* input_data = input_layer->mutable_cpu_data();
    
    unsigned long img_number = imgs.size();
    unsigned long img_channel = imgs[0].size();
    unsigned long img_height = imgs[0][0].rows();
    unsigned long img_width = imgs[0][0].cols();
    
    unsigned long index = 0;
    for (int i = 0; i < img_number; i++)
    {
        for (int c = 0; c < img_channel; c++)
        {
            for (int h = 0; h < img_height; h++)
            {
                for (int w = 0; w < img_width; w++)
                {
                    *(input_data + index) = imgs[i][c](h, w);
                    index++;
                }
            }
        }
    }
}
#endif /* data_transformer_hpp */
