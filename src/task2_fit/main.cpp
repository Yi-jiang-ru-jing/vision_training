#include <cstddef>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>
#include <ceres/ceres.h>
#include <random>
#include <string>
#include <algorithm>


int check_video(const cv::VideoCapture& vid);
double identify_calculate(cv::Mat& frame, double center_x, double center_y);
cv::Mat identify_target(cv::Mat& frame);
double calculate_omega(double angle1, double angle2, int FPS);
void fit_omega(std::vector<double>& time, std::vector<double>& omega, double* params_result);
double residual_solve(std::vector<double>& time, std::vector<double>& omegas, const double* params_result, std::vector<double>& residuals, std::vector<double>& omega_pre);
void drawPlot(const std::vector<double>& time, 
              const std::vector<double>& data1, 
              const std::vector<double>& data2, 
              const std::string& title, 
              const std::string& y_label, 
              const std::string& save_path, 
              double max_val, double min_val, double data3);
struct CostFunctor{
    CostFunctor(double t, double Omega) : t_(t), Omega_(Omega) {}
    template <typename T>
    bool operator()(const T* const params, T* residual) const{
        T A = params[0];
        T b = params[1];
        T omega = params[2];
        T phi = params[3];
        T Omega_pre= A * ceres::sin(omega * T(t_) + phi) + b;
        residual[0] = T(Omega_) - Omega_pre;
        return true;
    }
    double t_;
    double Omega_;
};



int main(){
    cv::VideoCapture vid("resources/task_2.mp4");
    if(check_video(vid) != 0) return 1;

    int FPS = 60;
    double center_x = 480.0;
    double center_y = 360.0;
    int radius = 220;
    int first = 0;
    std::vector<double> angles;
    std::vector<double> omegas;
    std::vector<double> time;
    int frame_width = static_cast<int>(vid.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(vid.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = FPS;
    cv::VideoWriter video_writer("result/task2_fit/tracking_overlay.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps, cv::Size(frame_width, frame_height));
    while(true){
        cv::Mat frame;
        vid >> frame;
        if(frame.empty()) break;
        // 识别与计算角度
        double angle = identify_calculate(frame, center_x, center_y); 
        angles.push_back(angle);
        time.push_back(time.size() / static_cast<double>(FPS));
        video_writer.write(frame); 
        cv::imshow("Video Frame", frame);
        if(cv::waitKey(30) == 27) break; 

    }
    int count = 0;
    for(int i=1; i < angles.size(); i++)
    {
        if(angles[i] <= 0 && angles[i-1]-(2*count*CV_PI) >= 0) count++;
        angles[i] = angles[i] + 2 * count * CV_PI;
    }
    
    // 计算角速度
    for(int i = 1; i < angles.size(); ++i){
        double omega = calculate_omega(angles[i-1], angles[i], FPS);
        omegas.push_back(omega);
    }
    // 拟合角速度
    double params_result[4];
    fit_omega(time, omegas, params_result);
    std::vector<double> omega_pre;
    std::vector<double> residuals;
    double rmse = residual_solve(time, omegas, params_result, residuals, omega_pre);
    drawPlot(time, omegas, omega_pre, "Angular Velocity Fitting Comparison", "Angular Velocity (rad/s)", "result/task2_fit/fit_comparison.png", 3.0, 0, rmse);
    std::vector<double> empty_data; 
    drawPlot(time, residuals, empty_data, 
         "Residuals", 
         "Residual (rad/s)", 
         "result/task2_fit/residuals.png", 0.3, -0.3, rmse);
    cv::destroyAllWindows();
    return 0;
}


// 检测视频
int check_video(const cv::VideoCapture& vid){
    if(!vid.isOpened()){
        std::cout << "Could not read the video: " << std::endl;
        return -1;
    }
    return 0;
}


// 识别与计算
double identify_calculate(cv::Mat& frame, double center_x, double center_y){
    cv::Mat mask = identify_target(frame);
    if(mask.empty()) return -1;
    // 计算质心
    cv::Moments moments = cv::moments(mask, true);
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;
    cv::circle(frame, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(frame, "Target", cv::Point(cx + 10, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::imshow("target", frame);
    //cv::waitKey(0);
    double angle = std::atan2(center_y - cy, cx - center_x);
    cv::imwrite("result/task2_images/blue_mask.png", mask);
    return angle;
}


// 识别青色目标
cv::Mat identify_target(cv::Mat& frame){
    cv::Mat hsv_frame, mask;
    cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_frame, cv::Scalar(78, 100, 100), cv::Scalar(130, 255, 255), mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::imshow("Blue Mask", mask);
    if(contours.empty()) return cv::Mat();
    cv::imshow("Blue Mask", mask);
    //cv::waitKey(0);
    return mask;
}


// 计算角速度
double calculate_omega(double angle1, double angle2, int FPS = 60){
    double delta_angle = angle2 - angle1;
    double omega = delta_angle / (1.0 / FPS);
    return omega;
}


// 拟合角速度
void fit_omega(std::vector<double>& time, std::vector<double>& omega, double* params_result){
    double A=0.7, b=1.4, omega_param = 1.8, phi = 0.5;
    double params[4] = {A, b, omega_param, phi};
    ceres::Problem problem;
    for(size_t i = 0; i < time.size(); ++i){
        auto* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 4>(new CostFunctor(time[i], omega[i]));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
    problem.SetParameterLowerBound(params, 0, 0.0); 
    problem.SetParameterLowerBound(params, 1, params[0]); 
    problem.SetParameterLowerBound(params, 2, 0.0);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Estimated parameters: A = " << params[0] << ", b = " << params[1] << ", omega = " << params[2] << ", phi = " << params[3] << std::endl;
    params_result[0] = params[0];
    params_result[1] = params[1];
    params_result[2] = params[2];
    params_result[3] = params[3];
}


// 残差求解
double residual_solve(std::vector<double>& time, std::vector<double>& omegas, const double* params_result, std::vector<double>& residuals, std::vector<double>& omega_pre){
    double A = params_result[0];
    double b = params_result[1];
    double omega = params_result[2];
    double phi = params_result[3];
    double rmse = 0.0;
    for(std::size_t i = 0; i < omegas.size(); ++i){
        double omega_cal = A * std::sin(omega * time[i] + phi) + b;
        omega_pre.push_back(omega_cal);
        double residual = omegas[i] - omega_pre[i];
        residuals.push_back(residual);
        rmse += residual * residual;
    }
    rmse = std::sqrt(rmse / (time.size()-1));
    return rmse;
}



void drawPlot(const std::vector<double>& time, 
              const std::vector<double>& data1, 
              const std::vector<double>& data2, 
              const std::string& title, 
              const std::string& y_label, 
              const std::string& save_path, double max_val, double min_val, double data3) {
    
    // 1. 画布尺寸
    int width = 1200, height = 600;
    int margin = 80;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 2. 数据范围（避免除以0）
    double min_t = time.front(), max_t = time.back();
    if (max_t == min_t) max_t = min_t + 1.0;
    double val_range = max_val - min_val;

    // 3. 映射函数
    auto mapX = [&](double t) { return margin + (t - min_t) / (max_t - min_t) * (width - 2 * margin); };
    auto mapY = [&](double v) { return margin + (1.0 - (v - min_val) / val_range) * (height - 2 * margin); };

    // 4. 绘制坐标轴与框线
    cv::rectangle(canvas, cv::Point(margin, margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 1);
    // 绘制0线（若包含0且跨越0线）
    if (min_val < 0 && max_val > 0) {
        int y_zero = (int)mapY(0);
        cv::line(canvas, cv::Point(margin, y_zero), cv::Point(width - margin, y_zero), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // 5. 绘制数据1（观测点，蓝色）
    for (size_t i = 0; i < time.size() && i < data1.size(); ++i) {
        if(std::isfinite(data1[i]))
            cv::circle(canvas, cv::Point((int)mapX(time[i]), (int)mapY(data1[i])), 2, cv::Scalar(255, 0, 0), -1);
    }

    // 6. 绘制数据2（拟合曲线，红色）
    if (!data2.empty()) {
        for (size_t i = 0; i < time.size() - 1 && i < data2.size() - 1; ++i) {
            cv::Point p1((int)mapX(time[i]), (int)mapY(data2[i]));
            cv::Point p2((int)mapX(time[i+1]), (int)mapY(data2[i+1]));
            cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
    int legend_y = margin + 20;
    // 7. 添加文字说明
    cv::putText(canvas, title, cv::Point(width/2 - 150, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    cv::putText(canvas, y_label, cv::Point(15, margin - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "Time (s)", cv::Point(width/2 - 40, height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    if(data2.empty()){
        char rmse_text[50];
        snprintf(rmse_text, sizeof(rmse_text), "RMSE: %.4f", data3); 
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(rmse_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        cv::putText(canvas, rmse_text, cv::Point(width-margin-text_size.width-5, legend_y-30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
        // 图例 (Top right)

    if (!data2.empty()) {
        cv::circle(canvas, cv::Point(width - margin - 180, legend_y), 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(canvas, "Observed", cv::Point(width - margin - 160, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::line(canvas, cv::Point(width - margin - 180, legend_y + 25), cv::Point(width - margin - 150, legend_y + 25), cv::Scalar(0, 0, 255), 2);
        cv::putText(canvas, "Fitted", cv::Point(width - margin - 140, legend_y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    } else {
        cv::circle(canvas, cv::Point(width - margin - 180, legend_y), 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(canvas, "Residual", cv::Point(width - margin - 160, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    }

    // 8. 保存图片
    cv::imshow("canvas", canvas);
    cv::waitKey(0);
    cv::imwrite(save_path, canvas);
    std::cout << "图片已保存: " << save_path << std::endl;
}