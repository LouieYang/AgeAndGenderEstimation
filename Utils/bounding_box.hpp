#ifndef bounding_box_hpp
#define bounding_box_hpp

#include <vector>

#include <opencv2/opencv.hpp>

/// @Brief: A very small class implementing bounding box instead of using
///         cv::Rect
class BoundingBox {
    /**
     *  @Brief: non-maxium suppression algorithm with the result averaged
     *
     *  @param std::vector<BoundingBox>& Raw bounding box being processed
     *  @param std::vector<BoundingBox>& Processed bounding box
     *  @param float                     The threshold of overlapping
     */
    friend void nms_average(std::vector<BoundingBox>&,
                            std::vector<BoundingBox>&, float);
    
    /**
     *  @Brief: non-maxium suppression algorithm with the result maximized
     *
     *  @param std::vector<BoundingBox>& Raw bounding box being processed
     *  @param std::vector<BoundingBox>& Processed bounding box
     *  @param float                     The threshold of overlapping
     */
    friend void nms_max(std::vector<BoundingBox>&,
                        std::vector<BoundingBox>&, float);
    
    /**
     *  @Brief: sort the bounding box by their probability with ">"
     *
     *  @param BoundingBox& first bounding box
     *  @param BoundingBox& second bounding box
     *
     *  @return bool
     */
    friend bool sort_by_confidence_reverse(const BoundingBox&,
                                           const BoundingBox&);
    
    /**
     *  @Brief: sort the bounding box by their size with "<"
     *
     *  @param BoundingBox& first bounding box
     *  @param BoundingBox& second bounding box
     *
     *  @return bool
     */
    friend bool sort_by_size(const BoundingBox&,
                             const BoundingBox&);
public:
    BoundingBox(float x, float y, float width, float height, float prob): x(x), y(y), width(width), height(height), prob(prob) {};
    BoundingBox(BoundingBox&&) = default;
    BoundingBox(const BoundingBox&) = default;
    
    BoundingBox& operator=(const BoundingBox&) = default;
    BoundingBox& operator=(BoundingBox&&) = default;
    BoundingBox() = default;
    
    ~BoundingBox() = default;
    
    /**
     *  @Brief: Change the bounding box to cv::Rect ignoring the prob
     */
    cv::Rect transformToCVRect() { return cv::Rect(x, y, width, height); };

    float getX() const  {  return x;   };
    float getY() const  {  return y;   };
    float getWidth() const  {  return width;   };
    float getHeight() const { return height;  };
    float getProb() const   {   return prob;    };
    
private:
    
    float area() const {  return width * height;  };
    
    float x;
    float y;
    float width;
    float height;
    float prob; /// The possibility of this area being a face
};

bool sort_by_confidence_reverse(const BoundingBox& a,
                                const BoundingBox& b);
bool sort_by_size(const BoundingBox&, const BoundingBox&);

void nms_average(std::vector<BoundingBox>&,
                 std::vector<BoundingBox>&, float);

void nms_max(std::vector<BoundingBox>&,
             std::vector<BoundingBox>&, float);

#endif /* bounding_box_hpp */