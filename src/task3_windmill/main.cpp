#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace DetectionParams {
constexpr int RED_HUE_LOW_MAX = 15;
constexpr int RED_HUE_HIGH_MIN = 168;
constexpr int MIN_SATURATION = 70;
//constexpr int MIN_VALUE = 65;
constexpr int MORPH_KERNEL_SIZE = 3;

constexpr double TARGET_MIN_AREA = 250.0;
constexpr double TARGET_MAX_AREA = 50000.0;
constexpr double TARGET_MIN_CIRCULARITY = 0.65;
constexpr float TARGET_MIN_RADIUS = 12.0f;
constexpr float TARGET_MAX_RADIUS = 220.0f;
constexpr double TARGET_MAX_CENTER_DISTANCE = 18.0;
constexpr double TARGET_MIN_INNER_AREA = 40.0;
constexpr double TARGET_MIN_INNER_CIRCULARITY = 0.45;
constexpr double TARGET_MIN_INNER_RADIUS_RATIO = 0.10;
constexpr double TARGET_MAX_INNER_RADIUS_RATIO = 0.85;
constexpr double TARGET_MAX_INNER_CENTER_OFFSET_RATIO = 0.20;

constexpr double R_MIN_AREA = 30.0;
constexpr double R_MAX_AREA = 2500.0;
constexpr double R_MIN_ASPECT_RATIO = 0.25;
constexpr double R_MAX_ASPECT_RATIO = 3.5;
constexpr double R_MIN_EXTENT = 0.15;
constexpr double R_MAX_EXTENT = 0.90;
constexpr size_t R_MIN_APPROX_VERTICES = 5;
constexpr double R_TEMPLATE_MATCH_THRESHOLD = 1.0;
}

struct DetectedTarget {
    cv::Point2f center{-1.0f, -1.0f};
    float radius = 0.0f;
};

struct TrackedTarget {
    int id = -1;
    double last_radius = 0.0;
    bool is_tracking = false;
    cv::Point2f last_center{-1.0f, -1.0f};
    int lost_frames = 0;
    std::string status = "lost";
};

constexpr int MAX_LOST_FRAMES = 15;
constexpr double MAX_REACQUIRE_DISTANCE = 180.0;

int check_video(const cv::VideoCapture& vid);
cv::Mat create_red_mask(const cv::Mat& frame, const int min_value);
std::vector<DetectedTarget> detect_target_candidates(const cv::Mat& red_mask, std::vector<cv::Rect>* target_boxes = nullptr);
std::optional<DetectedTarget> detect_r_marker(const cv::Mat& red_mask,
                                           const std::vector<cv::Rect>& excluded_boxes,
                                           const cv::Point2f& previous_center = cv::Point2f(-1.0f, -1.0f));
void update_tracking_state(TrackedTarget& target, const std::optional<DetectedTarget>& new_target);
double match_r_template(const std::vector<cv::Point>& contour);
std::optional<DetectedTarget> select_tracking_target(const std::vector<DetectedTarget>& targets,
                                                    const TrackedTarget& tracked_target);
cv::Point2f get_frame_center(const cv::Mat& frame);
cv::Point2f choose_target_center(const cv::Mat& frame,
                               const std::vector<DetectedTarget>& targets,
                               const TrackedTarget& tracked_target);
void draw_tracking_result(cv::Mat& frame,
                         const std::vector<DetectedTarget>& targets,
                         const std::optional<DetectedTarget>& r_marker,
                         const TrackedTarget& tracked_target);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [output_dir]" << std::endl;
        return 1;
    }

    cv::VideoCapture video(argv[1]);
    if (check_video(video) != 0) {
        return 1;
    }

    const std::filesystem::path output_dir =
        argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path("result/task3_windmill");
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
        std::cerr << "Could not create output directory: " << output_dir << std::endl;
        return 1;
    }

    const double fps = video.get(cv::CAP_PROP_FPS) > 0.0
        ? video.get(cv::CAP_PROP_FPS)
        : 30.0;
    const int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    const cv::Size frame_size(frame_width, frame_height);
    const std::string codec = "mp4v";

    cv::VideoWriter binary_writer(
        (output_dir / "binary_video.mp4").string(),
        cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]),
        fps,
        frame_size,
        false);
    cv::VideoWriter marked_writer(
        (output_dir / "recognition_video.mp4").string(),
        cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]),
        fps,
        frame_size,
        true);

    if (!binary_writer.isOpened() || !marked_writer.isOpened()) {
        std::cerr << "Could not open output video writers in: " << output_dir << std::endl;
        return 1;
    }

    TrackedTarget tracked_target;
    tracked_target.is_tracking = false;

    while (true) {
        cv::Mat frame;
        video >> frame;
        if (frame.empty()) {
            break;
        }

        cv::Mat red_mask_R = create_red_mask(frame, 65);
        cv::Mat red_mask_target = create_red_mask(frame, 50);
        std::vector<cv::Rect> target_boxes_target, target_boxes_R;
        std::vector<DetectedTarget> targets_target = detect_target_candidates(red_mask_target, &target_boxes_target);

        std::optional<DetectedTarget> best_target = select_tracking_target(targets_target, tracked_target);
        std::vector<DetectedTarget> targets_R = detect_target_candidates(red_mask_R, &target_boxes_R);

        std::optional<DetectedTarget> r_marker = detect_r_marker(red_mask_R, target_boxes_R, tracked_target.last_center);
        update_tracking_state(tracked_target, best_target);
        draw_tracking_result(frame, targets_target, r_marker, tracked_target);

        binary_writer.write(red_mask_R);
        marked_writer.write(frame);

        cv::imshow("video", frame);
        cv::imshow("red_mask", red_mask_R);
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

int check_video(const cv::VideoCapture& vid) {
    if (!vid.isOpened()) {
        std::cout << "Could not read the video." << std::endl;
        return -1;
    }
    return 0;
}

cv::Mat create_red_mask(const cv::Mat& frame, const int min_value) {
    cv::Mat hsv_frame;
    cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

    cv::Mat mask_low, mask_high, red_mask;
    cv::inRange(hsv_frame,
                cv::Scalar(0, DetectionParams::MIN_SATURATION, min_value),
                cv::Scalar(DetectionParams::RED_HUE_LOW_MAX, 255, 255),
                mask_low);
    cv::inRange(hsv_frame,
                cv::Scalar(DetectionParams::RED_HUE_HIGH_MIN,
                           DetectionParams::MIN_SATURATION,
                           min_value),
                cv::Scalar(180, 255, 255),
                mask_high);
    cv::add(mask_low, mask_high, red_mask);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(DetectionParams::MORPH_KERNEL_SIZE,
                 DetectionParams::MORPH_KERNEL_SIZE));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
    return red_mask;
}

std::vector<DetectedTarget> detect_target_candidates(const cv::Mat& red_mask, std::vector<cv::Rect>* target_boxes) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(red_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<DetectedTarget> targets;
    if (target_boxes != nullptr) {
        target_boxes->clear();
    }

    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        double area = cv::contourArea(contour);
        if (area < DetectionParams::TARGET_MIN_AREA ||
            area > DetectionParams::TARGET_MAX_AREA) {
            continue;
        }

        double perimeter = cv::arcLength(contour, true);
        if (perimeter < 1e-6) {
            continue;
        }

        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity <= DetectionParams::TARGET_MIN_CIRCULARITY) {
            continue;
        }

        cv::Point2f center;
        float radius = 0.0f;
        cv::minEnclosingCircle(contour, center, radius);
        if (radius <= DetectionParams::TARGET_MIN_RADIUS ||
            radius > DetectionParams::TARGET_MAX_RADIUS) {
            continue;
        }

        bool has_concentric_pattern = false;
        for (int child_idx = hierarchy[i][2];
             child_idx != -1;
             child_idx = hierarchy[child_idx][0]) {
            const auto& child_contour = contours[child_idx];
            double child_area = cv::contourArea(child_contour);
            if (child_area < DetectionParams::TARGET_MIN_INNER_AREA) {
                continue;
            }

            double child_perimeter = cv::arcLength(child_contour, true);
            if (child_perimeter < 1e-6) {
                continue;
            }

            double child_circularity =
                4.0 * CV_PI * child_area / (child_perimeter * child_perimeter);
            if (child_circularity < DetectionParams::TARGET_MIN_INNER_CIRCULARITY) {
                continue;
            }

            cv::Point2f child_center;
            float child_radius = 0.0f;
            cv::minEnclosingCircle(child_contour, child_center, child_radius);
            if (child_radius <= 0.0f) {
                continue;
            }

            double center_dist = cv::norm(center - child_center);
            double radius_ratio = static_cast<double>(child_radius) / radius;
            double max_center_offset = std::max(
                DetectionParams::TARGET_MAX_CENTER_DISTANCE,
                radius * DetectionParams::TARGET_MAX_INNER_CENTER_OFFSET_RATIO);

            if (center_dist <= max_center_offset &&
                radius_ratio >= DetectionParams::TARGET_MIN_INNER_RADIUS_RATIO &&
                radius_ratio <= DetectionParams::TARGET_MAX_INNER_RADIUS_RATIO) {
                has_concentric_pattern = true;
                break;
            }
        }

        if (!has_concentric_pattern) {
            continue;
        }

        targets.push_back({center, radius});
        if (target_boxes != nullptr) {
            target_boxes->push_back(cv::boundingRect(contour));
        }
    }

    return targets;
}

std::optional<DetectedTarget> detect_r_marker(const cv::Mat& red_mask,
                                            const std::vector<cv::Rect>& excluded_boxes,
                                            const cv::Point2f& previous_center) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::optional<DetectedTarget> best_marker;
    double best_score = std::numeric_limits<double>::max();

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < DetectionParams::R_MIN_AREA ||
            area > DetectionParams::R_MAX_AREA) {
            continue;
        }

        cv::Point2f center;
        float radius = 0.0f;
        cv::minEnclosingCircle(contour, center, radius);

        bool inside_target = false;
        for (const auto& box : excluded_boxes) {
            cv::Rect padded_box = box + cv::Size(20, 20) - cv::Point(10, 10);
            if (padded_box.contains(center)) {
                inside_target = true;
                break;
            }
        }
        if (inside_target) {
            continue;
        }

        cv::Rect rect = cv::boundingRect(contour);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        float aspect_ratio = static_cast<float>(rect.width) / static_cast<float>(rect.height);
        double extent = area / static_cast<double>(rect.width * rect.height);
        double perimeter = cv::arcLength(contour, true);
        if (perimeter < 1e-6) {
            continue;
        }

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        bool is_roughly_linear = std::abs(rect.width - rect.height) > std::max(rect.width, rect.height) * 0.8;
        if ((aspect_ratio < DetectionParams::R_MIN_ASPECT_RATIO ||
             aspect_ratio > DetectionParams::R_MAX_ASPECT_RATIO) ||
            (extent < DetectionParams::R_MIN_EXTENT ||
             extent > DetectionParams::R_MAX_EXTENT) ||
            (approx.size() <= DetectionParams::R_MIN_APPROX_VERTICES) ||
            is_roughly_linear) {
            continue;
        }

        const double template_score = match_r_template(contour);
        if (template_score > DetectionParams::R_TEMPLATE_MATCH_THRESHOLD) {
            continue;
        }

        double score = previous_center.x >= 0.0f
            ? cv::norm(center - previous_center) + template_score * 25.0
            : -area + template_score * 25.0;

        if (score < best_score) {
            best_score = score;
            best_marker = DetectedTarget{center, radius};
        }
    }

    return best_marker;
}

double match_r_template(const std::vector<cv::Point>& contour) {
    static const std::vector<cv::Point> r_template = [] {
        cv::Mat template_mask = cv::Mat::zeros(160, 160, CV_8UC1);
        cv::putText(template_mask, "R", cv::Point(25, 125),
                    cv::FONT_HERSHEY_SIMPLEX, 3.5, cv::Scalar(255), 8,
                    cv::LINE_AA);

        std::vector<std::vector<cv::Point>> template_contours;
        cv::findContours(template_mask, template_contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        if (template_contours.empty()) {
            return std::vector<cv::Point>{};
        }

        size_t largest_index = 0;
        for (size_t i = 1; i < template_contours.size(); ++i) {
            if (cv::contourArea(template_contours[i]) >
                cv::contourArea(template_contours[largest_index])) {
                largest_index = i;
            }
        }
        return template_contours[largest_index];
    }();

    if (r_template.empty() || contour.empty()) {
        return std::numeric_limits<double>::max();
    }

    return cv::matchShapes(contour, r_template, cv::CONTOURS_MATCH_I1, 0.0);
}

void update_tracking_state(TrackedTarget& target, const std::optional<DetectedTarget>& new_target) {
    if (!new_target.has_value()) {
        ++target.lost_frames;
        if (target.lost_frames <= MAX_LOST_FRAMES && target.is_tracking) {
            // Keep the last valid center during a short occlusion or missed detection.
            target.status = "track_lost";
            return;
        }

        if (target.lost_frames > MAX_LOST_FRAMES) {
            target.is_tracking = false;
            target.status = "lost";
            target.id = -1;
            target.last_center = cv::Point2f(-1.0f, -1.0f);
            target.last_radius = 0.0;
        }
        return;
    }

    target.last_center = new_target->center;
    target.last_radius = new_target->radius;
    target.lost_frames = 0;
    target.is_tracking = true;
    target.status = "detect";
    if (target.id == -1) {
        target.id = 1;
    }
}

std::optional<DetectedTarget> select_tracking_target(const std::vector<DetectedTarget>& targets,
                                                    const TrackedTarget& tracked_target) {
    if (targets.empty()) {
        return std::nullopt;
    }

    if (!tracked_target.is_tracking || tracked_target.last_center.x < 0.0f) {
        return targets.front();
    }

    std::optional<DetectedTarget> best_candidate;
    double best_distance = std::numeric_limits<double>::max();
    for (const auto& candidate : targets) {
        const double distance = cv::norm(candidate.center - tracked_target.last_center);
        if (distance < best_distance) {
            best_distance = distance;
            best_candidate = candidate;
        }
    }

    if (best_candidate.has_value() && best_distance <= MAX_REACQUIRE_DISTANCE) {
        return best_candidate;
    }

    return std::nullopt;
}

cv::Point2f get_frame_center(const cv::Mat& frame) {
    return cv::Point2f(static_cast<float>(frame.cols) * 0.5f,
                       static_cast<float>(frame.rows) * 0.5f);
}

cv::Point2f choose_target_center(const cv::Mat& frame,
                               const std::vector<DetectedTarget>& targets,
                               const TrackedTarget& tracked_target) {
    if (tracked_target.is_tracking && tracked_target.last_center.x >= 0.0f) {
        return tracked_target.last_center;
    }
    if (!targets.empty()) {
        return targets.front().center;
    }
    return get_frame_center(frame);
}

void draw_tracking_result(cv::Mat& frame,
                         const std::vector<DetectedTarget>& targets,
                         const std::optional<DetectedTarget>& r_marker,
                         const TrackedTarget& tracked_target) {
    for (const auto& target : targets) {
        cv::circle(frame, target.center, static_cast<int>(target.radius), cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, target.center, 4, cv::Scalar(0, 255, 0), -1);
    }

    const bool has_target = (!targets.empty()) || (tracked_target.is_tracking && tracked_target.last_center.x >= 0.0f);
    if (r_marker.has_value() && has_target) {
        const cv::Point2f& r_center = r_marker->center;
        cv::circle(frame, r_center, 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(frame, "R", cv::Point(r_center.x + 10, r_center.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::Point2f target_center = choose_target_center(frame, targets, tracked_target);
        cv::line(frame, r_center, target_center, cv::Scalar(255, 0, 0), 2);
        double angle = std::atan2(r_center.y - target_center.y,
                                 target_center.x - r_center.x) * 180.0 / CV_PI;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << angle << " deg";
        cv::putText(frame, ss.str(), cv::Point(target_center.x + 15, target_center.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }

    if (tracked_target.is_tracking && tracked_target.last_center.x >= 0.0f) {
        cv::circle(frame, tracked_target.last_center, 6, cv::Scalar(255, 255, 0), 2);
        std::stringstream status_text;
        status_text << "ID:" << tracked_target.id << " " << tracked_target.status;
        cv::putText(frame, status_text.str(), cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    }
}
