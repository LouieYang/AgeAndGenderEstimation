#include "gender_estimation.hpp"


std::vector<Gender> gender_estimation(const cv::Mat& Image, const std::vector<BoundingBox>& face_rois)
{
    std::vector<Gender> genders;
    
    std::vector<float> mean(SetMean(project_root + "mean.binaryproto", 3));

    std::shared_ptr<caffe::Net<double>> net(new caffe::Net<double>(project_root + "deploy_gender2.prototxt", caffe::Phase::TEST));
    net->CopyTrainedLayersFrom(project_root + "gender_net.caffemodel");
    
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
        double predictMan = *(predicts);
        ++predicts;
        double predictWoman = *(predicts);
                
        if (predictWoman > predictMan)
        {
            genders.push_back(Gender::Woman);
        }
        else if (predictMan > predictWoman)
        {
            genders.push_back(Gender::Man);
        }
        else
        {
            genders.push_back(Gender::Error);
        }
    }
    
    return std::move(genders);
}
