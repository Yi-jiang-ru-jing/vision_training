#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>


int check_image(const cv::Mat& img, const cv::String& image_path);
int output_gray_image(const cv::Mat& img);
int output_hsv_channels(const cv::Mat& img);
int Blur(const cv::Mat& img);
cv::Mat red_extract(const cv::Mat& img);
int morphology(const cv::Mat& mask);
int draw_and_transform(const cv::Mat& img);


int main(){
    cv::String image_path = "resources/test_image.jpg";
    cv::Mat img = cv::imread(image_path);
    // 检验读图，输出灰度图和H,S,V单通道图
    check_image(img, image_path);
    output_gray_image(img);
    output_hsv_channels(img);
    //实现三种滤波
    Blur(img);
    //红色提取
    cv::Mat red_mask = red_extract(img);
    //形态学操作
    morphology(red_mask);
    //绘制与变换
    draw_and_transform(img);
    cv::waitKey(0);
    return 0;
}


// 检测读图是否成功
int check_image(const cv::Mat& img, const cv::String& image_path){ 
    if(img.empty()){
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}


// 输出灰度图
int output_gray_image(const cv::Mat& img){
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::imshow("Gray Image", gray_img);
    cv::imwrite("result/task1_images/gray.png", gray_img);
    cv::waitKey(0);
    return 0;
}


// 输出H,S,V单通道图
int output_hsv_channels(const cv::Mat& img){
    cv::Mat hsv_img;
    cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_img, hsv_channels);
    cv::imshow("H Image", hsv_channels[0]);
    cv::imwrite("result/task1_images/hsv_h.png", hsv_channels[0]);
    cv::imshow("S Image", hsv_channels[1]);
    cv::imwrite("result/task1_images/hsv_s.png", hsv_channels[1]);
    cv::imshow("V Image", hsv_channels[2]);
    cv::imwrite("result/task1_images/hsv_v.png", hsv_channels[2]);
    cv::waitKey(0);
    return 0;
}


// 实现三种滤波
int Blur(const cv::Mat& img){
    cv::Mat meanImg, gaussImg, medianImg;
    cv::blur(img, meanImg, cv::Size(5, 5));
    cv::GaussianBlur(img, gaussImg, cv::Size(5, 5), 0);
    cv::medianBlur(img, medianImg, 5);
    cv::imshow("Mean", meanImg);
    cv::imwrite("result/task1_images/mean_filter.png", meanImg);
    cv::imshow("Gaussian", gaussImg);
    cv::imwrite("result/task1_images/gauss_filter.png", gaussImg);
    cv::imshow("Median", medianImg);
    cv::imwrite("result/task1_images/median_filter.png", medianImg);
    cv::waitKey(0);
    return 0;
}


// 红色提取
cv::Mat red_extract(const cv::Mat& img){
    cv::Mat hsv_img, maskLow, maskHigh, mask;
    cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_img, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), maskLow);
    cv::inRange(hsv_img, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), maskHigh);
    cv::add(maskLow, maskHigh, mask);
    cv::imshow("Red Mask", mask);
    cv::waitKey(0);

    cv::imwrite("result/task1_images/red_mask.png", mask);
    return cv::Mat(mask);
}


// 形态学操作
int morphology(const cv::Mat& mask){
    cv::Mat eroded, dilated, opened, closed;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(mask, eroded, kernel);
    cv::dilate(mask, dilated, kernel);
    cv::morphologyEx(mask, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, kernel);
    cv::imshow("Eroded", eroded);
    cv::imwrite("result/task1_images/erode.png", eroded);
    cv::imshow("Dilated", dilated);
    cv::imwrite("result/task1_images/dilate.png", dilated);
    cv::imshow("Opened", opened);
    cv::imwrite("result/task1_images/open.png", opened);
    cv::imshow("Closed", closed);
    cv::imwrite("result/task1_images/close.png", closed);
    cv::waitKey(0);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(closed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contour_img = cv::Mat::zeros(closed.size(), CV_8UC3);
    drawContours(contour_img, contours, -1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Contours", contour_img);
    // 轮廓筛选与面积计算
    cv::Mat result = contour_img.clone();
    for(size_t i=0; i<contours.size(); i++){
        double area = cv::contourArea(contours[i]);
        if(area < 1000.0) continue;

        cv::Rect box = cv::boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if(ratio < 0.2 || ratio > 5.0) continue;
        // 多边形近似
        std::vector<std::vector<cv::Point>> approx;
        double perimeter = cv::arcLength(contours[i], true);
        double epsilon = 0.001 * perimeter;
        cv::approxPolyDP(contours[i], approx.emplace_back(), epsilon, true);
        if(approx.back().size() < 4) continue;
        std::vector<std::vector<cv::Point>> approx_contours = {approx};
        cv::polylines(result, approx_contours, true, cv::Scalar(0, 255, 0), 2);

        std::cout << "Contour " << i << ": Area = " << area << std::endl;
        cv::putText(result, "Area: " + std::to_string(static_cast<int>(area)), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        cv::rectangle(result, box, cv::Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i), cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("Filtered Contours", result);
    cv::imwrite("result/task1_images/contour_boxes.png", result);
    cv::waitKey(0);
    return 0;
}


//绘制与变换
int draw_and_transform(const cv::Mat& img){
    const double pi = std::acos(-1.0);
    cv::Mat draw_img = img.clone();
    // 绘制
    cv::circle(draw_img, cv::Point(200, 200), 100, cv::Scalar(0, 255, 0), 2);
    cv::rectangle(draw_img, cv::Point(300, 300), cv::Point(500, 500), cv::Scalar(255, 0, 0), 2);
    cv::putText(draw_img, "Robomaster NB", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0), 1);
    cv::imshow("Drawn Image", draw_img);
    cv::imwrite("result/task1_images/draw.png", draw_img);
    // 旋转
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), 35, 1.0);
    cv::Mat rotated_img;
    cv::warpAffine(draw_img, rotated_img, rotation_matrix, draw_img.size());
    cv::imshow("Rotated Image", rotated_img);
    cv::imwrite("result/task1_images/rotated_35deg.png", rotated_img);
    // 裁减
    cv::Rect roi(0, 0, img.cols / 2, img.rows / 2);
    cv::Mat cropped_img = img(roi).clone();
    cv::imshow("Cropped Image", cropped_img);
    cv::imwrite("result/task1_images/crop_top_left.png", cropped_img);
    cv::waitKey(0);
    return 0;
}