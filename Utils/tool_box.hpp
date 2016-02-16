#ifndef tool_box_hpp
#define tool_box_hpp

#ifndef CPU_ONLY
#define CPU_ONLY
#endif

#include <caffe/caffe.hpp>
#include <map>

#include "bounding_box.hpp"
#include "data_transformer.hpp"

/**
 *  @Brief: Read the mean RGB values from mean.prototxt and use single
 *          value for convenience
 *
 *  @param std::string& mean.prototxt
 *  @param int          number of channels
 *
 *  @return A vector of BGR values
 */
std::vector<float> SetMean(const std::string&, int);

/**
 *  @Brief: Since the face extracted by face detection ignoring its hair
 *          and other important features, which, is essential for gender
 *          estimation and age estimation.
 *
 *  @param BoundingBox& face bounding box area
 *  @param int          image height
 *  @param int          image width
 *
 *  @return head bounding box area
 */
BoundingBox extend_face_to_whole_head(const BoundingBox&, const int, const int);

const int cellsize = 227;

/**
 *  @Brief: return the most N maximum value in the vector
 *
 *  @param v vector
 *  @param N the one user choose
 *
 *  @return most N big values index
 */
std::vector<int> Argmax(const std::vector<float>& v, int N);

#endif /* tool_box_hpp */
