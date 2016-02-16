#include "bounding_box.hpp"

void nms_average(std::vector<BoundingBox>& bd,
                 std::vector<BoundingBox>& processedBox,
                 float overlapped_thresh)
{
    std::sort(bd.begin(), bd.end(), sort_by_confidence_reverse);
    while (bd.size() != 0)
    {
        std::vector<int> iddlt(1, 0);
        
        float x11 = bd[0].x;
        float y11 = bd[0].y;
        float x12 = bd[0].x + bd[0].height;
        float y12 = bd[0].y + bd[0].width;
        
        if (bd.size() > 1)
        {
            for (int j = 1; j < bd.size(); j++)
            {
                float x21 = bd[j].x;
                float y21 = bd[j].y;
                float x22 = bd[j].x + bd[j].height;
                float y22 = bd[j].y + bd[j].width;
                
                float x_overlap = MAX(0, MIN(x12, x22) - MAX(x11, x21));
                float y_overlap = MAX(0, MIN(y12, y22) - MAX(y11, y21));
                
                if (x_overlap * y_overlap > MIN(bd[0].area(), bd[j].area()) * overlapped_thresh)
                {
                    iddlt.push_back(j);
                }
            }
        }
        
        float x_average  = 0;
        float y_average  = 0;
        float width      = 0;
        float height     = 0;
        float confidence = 0;
        
        for (int i = 0; i < iddlt.size(); i++)
        {
            x_average  += bd[iddlt[i]].x;
            y_average  += bd[iddlt[i]].y;
            width      += bd[iddlt[i]].width;
            height     += bd[iddlt[i]].height;
            confidence += bd[iddlt[i]].prob;
        }
        x_average /= iddlt.size();
        y_average /= iddlt.size();
        width /= iddlt.size();
        height /= iddlt.size();
        confidence /= iddlt.size();
        
        processedBox.emplace_back(BoundingBox(y_average, x_average, width, height, confidence));
        
        
        for (int i = 0; i < iddlt.size(); i++)
        {
            bd.erase(bd.begin() + iddlt[i] - i);
        }
    }
}

void nms_max(std::vector<BoundingBox>& bd,
             std::vector<BoundingBox>& processedBox,
             float overlapped_thresh)
{
    std::sort(bd.begin(), bd.end(), sort_by_size);
    for (int i = 0; i < bd.size(); i++)
    {
        int j = 0;
        for (; j < processedBox.size(); j++)
        {
            /* Calculate the overlapped area */
            float x11 = bd[i].x;
            float y11 = bd[i].y;
            float x12 = bd[i].x + bd[i].height;
            float y12 = bd[i].y + bd[i].width;
            
            float x21 = processedBox[j].x;
            float y21 = processedBox[j].y;
            float x22 = processedBox[j].x + processedBox[j].height;
            float y22 = processedBox[j].y + processedBox[j].width;
            
            float x_overlap = MAX(0, MIN(x12, x22) - MAX(x11, x21));
            float y_overlap = MAX(0, MIN(y12, y22) - MAX(y11, y21));
            
            if (x_overlap * y_overlap > MIN(bd[i].area(), processedBox[j].area()) * overlapped_thresh)
            {
                if (processedBox[j].prob < bd[i].prob)
                {
                    processedBox[j] = bd[i];
                }
                break;
            }
        }
        if (j == processedBox.size())
        {
            processedBox.emplace_back(bd[i]);
        }
    }
}

bool sort_by_confidence_reverse(const BoundingBox& a, const BoundingBox& b)
{
    return a.prob > b.prob;
}

bool sort_by_size(const BoundingBox& a, const BoundingBox& b)
{
    return a.area() < b.area();
}