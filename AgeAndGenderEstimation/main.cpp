/*******************************************************************
 *  Copyright(c) 2016
 *  All rights reserved.
 *
 *  Name: Age and Gender estimation
 *  Lib: OpenCV2, Eigen3, caffe
 *  Date: 2016-2-16
 *  Author: Yang
 ******************************************************************/

#include <iostream>

#include "face_detection.hpp"
#include "gender_estimation.hpp"
#include "age_estimation.hpp"

int main(int argc, const char * argv[]) {
    
    cv::Mat image = cv::imread(project_root + "Marianne_Stanley_0001.jpg");
    
    std::vector<BoundingBox> face_area;
    face_detection(image, face_area);
    
    std::vector<Gender> genders(gender_estimation(image, face_area));
    std::vector<Age> ages(age_estimation(image, face_area));
    
//    auto print_genders = [](Gender a)
//    {
//        if (a == Gender::Man)
//        {
//            std::cout << "Man" << std::endl;
//        }
//        else if (a == Gender::Woman)
//        {
//            std::cout << "Woman" << std::endl;
//        }
//        else
//        {
//            std::cout << "Error!\n" << "Nan value is detected" << std::endl;
//        }
//    };
//    
//    auto print_ages = [](Age a)
//    {
//        switch (a)
//        {
//            case Age::R0_2: std::cout << "Age From 0 to 2" << std::endl;
//                break;
//            case Age::R4_6: std::cout << "Age From 4 to 6" << std::endl;
//                break;
//            case Age::R8_13: std::cout << "Age From 8 to 13" << std::endl;
//                break;
//            case Age::R15_20: std::cout << "Age From 15 to 20" << std::endl;
//                break;
//            case Age::R25_32: std::cout << "Age From 25 to 32" << std::endl;
//                break;
//            case Age::R38_43: std::cout << "Age From 38 to 43" << std::endl;
//                break;
//            case Age::R48_53: std::cout << "Age From 48 to 53" << std::endl;
//                break;
//            case Age::R60_: std::cout << "Age over 60" << std::endl;
//                break;
//            default: std::cout << "Error occurs!" << std::endl;
//        }
//    };
//    
//    std::for_each(genders.begin(), genders.end(), print_genders);
//    std::for_each(ages.begin(), ages.end(), print_ages);
    
    auto i = 0;
    for (const auto& roi: face_area)
    {
        cv::Rect head(extend_face_to_whole_head(roi, image.rows, image.cols).transformToCVRect());

        cv::rectangle(image, head, cv::Scalar(255, 255, 255));

        cv::putText(image, gender_listss[genders[i]], cv::Point(head.x, head.y + 10), 0, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::putText(image, age_list[ages[i++]], cv::Point(head.x, head.y + head.height), 0, 0.7, cv::Scalar(255, 0, 0), 2);
    }
    
    cv::imshow("Ds", image);
    cv::waitKey();

    return 0;
}
