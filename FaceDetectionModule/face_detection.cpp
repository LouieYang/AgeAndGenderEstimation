#include "face_detection.hpp"

void face_detection(const cv::Mat& wildImage, std::vector<BoundingBox>& bdBoxes)
{
    int min_scale = 0;
    int max_scale = 0;
    int factor_count = 0;
    float factor = 0.7937;
    float delim = 0;
    
    std::vector<float> scalings;
    
    max_scale = MAX(wildImage.cols, wildImage.rows);
    min_scale = MIN(wildImage.cols, wildImage.rows);
    delim = MIN(2500.0 / max_scale, 5);
    
    while (delim > 1 + 1e-4)
    {
        scalings.push_back(delim);
        delim *= factor;
    }
    
    while (min_scale >= 227)
    {
        scalings.push_back(pow(factor, factor_count));
        min_scale *= factor;
    }
    
    std::vector<BoundingBox> TmpBoundingBox;
    
    
    std::vector<cv::Mat> channels;
    cv::split(wildImage, channels);
    
    for (const auto& scaling: scalings)
    {
        
        const int scaledCols = wildImage.cols * scaling;
        const int scaledRows = wildImage.rows * scaling;
        
        modify_prototxt(scaledRows, scaledCols);
        
        cv::Mat redChannel, greenChannel, blueChannel;
        cv::resize(channels[0], blueChannel,
                   cv::Size(scaledCols, scaledRows));
        cv::resize(channels[1], redChannel,
                   cv::Size(scaledCols, scaledRows));
        cv::resize(channels[2], greenChannel,
                   cv::Size(scaledCols, scaledRows));
        
        std::vector<Eigen::MatrixXf> rgbImage;
        rgbImage.emplace_back(OpenCV2Eigen(blueChannel));
        rgbImage.emplace_back(OpenCV2Eigen(greenChannel));
        rgbImage.emplace_back(OpenCV2Eigen(redChannel));
        
        rgbImage[0].array() -= red_channel_mean;
        rgbImage[1].array() -= green_channel_mean;
        rgbImage[2].array() -= blue_channel_mean;
        
        std::shared_ptr<caffe::Net<float>> net(new caffe::Net<float>(project_root + "face_full_conv2.prototxt", caffe::Phase::TEST));
        net->CopyTrainedLayersFrom(project_root + "face_full_conv.caffemodel");
        
        std::vector<std::vector<Eigen::MatrixXf>> rgbImages{std::move(rgbImage)};
        Eigen2Blob(rgbImages, net);
        net->ForwardPrefilled();
        
        caffe::Blob<float>* output_layer = net->output_blobs()[0];
        float* predict = const_cast<float*>(output_layer->cpu_data() + output_layer->shape(2) * output_layer->shape(3));
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> prob(predict, output_layer->shape(2), output_layer->shape(3));
        
        add_to_boundingbox(prob, scaling, TmpBoundingBox);
    }
    
    std::vector<BoundingBox> nms_averaged_bd;
    nms_average(TmpBoundingBox, nms_averaged_bd, 0.2);
    bdBoxes.clear();
    nms_max(nms_averaged_bd, bdBoxes, 0.3);
}

void modify_prototxt(const int rows, const int cols)
{
    std::ifstream protoIn(project_root + "face_full_conv.prototxt", std::ios::in);
    std::ofstream protoOut(project_root + "face_full_conv2.prototxt", std::ios::out);
    
    auto index = 0;
    for (std::string line; std::getline(protoIn, line); index++)
    {
        if (index == 5)
        {
            protoOut << "input_dim: " << rows << '\n';
        }
        else if (index == 6)
        {
            protoOut << "input_dim: " << cols << '\n';
        }
        else
        {
            protoOut << line << '\n';
        }
    }
    protoOut.close();
    protoIn.close();
}

void add_to_boundingbox(const Eigen::MatrixXf& prob, const float scale, std::vector<BoundingBox>& boundingBox)
{
    const int stride = 32;
    const int cell_size = 227;
    
    for (int h = 0; h < prob.rows(); ++h)
    {
        for (int w = 0; w < prob.cols(); ++w)
        {
            if (prob(h, w) >= 0.85)
            {
                boundingBox.emplace_back(BoundingBox(float(w * stride) / scale, float(h * stride) / scale, float(cell_size) / scale, float(cell_size) / scale, prob(h, w)));
            }
        }
    }
}
