#include "age_estimation.hpp"

std::vector<Age> age_estimation(const cv::Mat& Image, const std::vector<BoundingBox>& face_rois)
{
    std::vector<Age> ages;
    
    std::vector<float> mean(SetMean(project_root + "mean.binaryproto", 3));
    
    std::shared_ptr<caffe::Net<double>> net(new caffe::Net<double>(project_root + "deploy_age2.prototxt", caffe::Phase::TEST));
    net->CopyTrainedLayersFrom(project_root + "age_net.caffemodel");
    
    for (const auto& face_roi: face_rois)
    {
        BoundingBox headRoi(extend_face_to_whole_head(face_roi, Image.rows, Image.cols));
        cv::Mat head;
        
        Image(cv::Rect(headRoi.getX(), headRoi.getY(), headRoi.getWidth(), headRoi.getHeight())).copyTo(head);
        
        cv::resize(head, head, cv::Size(cellsize, cellsize));
        
        std::vector<cv::Mat> headCVChannels;
        
        cv::split(head, headCVChannels);
        
        std::vector<Eigen::MatrixXf> matChannels;
        matChannels.emplace_back(OpenCV2Eigen(headCVChannels[0]));
        matChannels.emplace_back(OpenCV2Eigen(headCVChannels[1]));
        matChannels.emplace_back(OpenCV2Eigen(headCVChannels[2]));
        
        matChannels[0].array() -= mean[0];
        matChannels[1].array() -= mean[1];
        matChannels[2].array() -= mean[2];
        
        std::vector<std::vector<Eigen::MatrixXf>> face_matrix{matChannels};
        
        Eigen2Blob<double>(face_matrix, net);
        net->ForwardPrefilled();
        caffe::Blob<double>* output_layer = net->output_blobs()[0];
        
        double* predicts = const_cast<double*>(output_layer->cpu_data());
        std::vector<float> possibilities(predicts, predicts + 8);

        switch(Argmax(possibilities, 1)[0])
        {
            case 0: ages.push_back(Age::R0_2); break;
            case 1: ages.push_back(Age::R4_6); break;
            case 2: ages.push_back(Age::R8_13); break;
            case 3: ages.push_back(Age::R15_20); break;
            case 4: ages.push_back(Age::R25_32); break;
            case 5: ages.push_back(Age::R38_43); break;
            case 6: ages.push_back(Age::R48_53); break;
            case 7: ages.push_back(Age::R60_); break;
            default: ages.push_back(Age::Error); break;
        }
    }
    
    return std::move(ages);
}