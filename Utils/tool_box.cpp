#include "tool_box.hpp"


std::vector<float> SetMean(const std::string& mean_file, int num_channels_)
{
    caffe::BlobProto blob_proto;
    
    caffe::ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    
    return std::vector<float>{channels[0].at<float>(0, 0),
        channels[1].at<float>(0, 0),
        channels[2].at<float>(0, 0)};
}

BoundingBox extend_face_to_whole_head(const BoundingBox& face,
                                      const int imageHeight, const int imageWidth)
{
    float extend = MIN(MIN(face.getX(), face.getY()), MIN(imageWidth - 1 - face.getX() - face.getWidth(), imageHeight - 1 - face.getY() - face.getHeight()));
    
    extend = MIN(extend, face.getHeight() / 2);
    
    return BoundingBox(face.getX() - extend, face.getY() - extend,
                       face.getWidth() + 2 * extend,
                       face.getHeight() + 2 * extend, face.getProb());
}

std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    auto PairCompare = [](const std::pair<float, int>& lhs,
                          const std::pair<float, int>& rhs)
    {
        return lhs.first > rhs.first;
    };
    
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    
    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}