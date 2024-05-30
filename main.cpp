#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 函数声明
void findBestMatch(Mat &result, Point &bestMatchLoc, double &bestMatchValue);

int main() {
    // 加载图像和模板
    Mat img = imread("../images/IMAGEB2.bmp", IMREAD_GRAYSCALE);
    Mat templ = imread("../images/pattern.bmp", IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        cerr << "Error: Cannot load images." << endl;
        return -1;
    }

    // 创建结果矩阵以及旋转后的模板
    Mat result(img.rows - templ.rows + 1, img.cols - templ.cols + 1, CV_32FC1);
    Mat rotatedTempl;

    // 记录最佳匹配的信息
    double bestMatchValue = -1;
    Point bestMatchLoc;
    double bestAngle;

    // 在不同角度下执行模板匹配
    for (int angle = 0; angle < 360; angle += 1) {
        // 旋转模板
        Mat rotationMatrix = getRotationMatrix2D(Point(templ.cols / 2, templ.rows / 2), angle, 1);
        warpAffine(templ, rotatedTempl, rotationMatrix, templ.size());

        // // 显示旋转后的模板
        // cv::imshow("Rotated Template", rotatedTempl);
        // cv::imshow("Original Image", img);
        // cv::waitKey(0);

        // 执行模板匹配
        matchTemplate(img, rotatedTempl, result, TM_CCORR_NORMED);

        // 找到最佳匹配位置和角度
        Point matchLoc;
        double matchValue;
        findBestMatch(result, matchLoc, matchValue);

        // 更新最佳匹配的信息
        if (matchValue > bestMatchValue) {
            bestMatchValue = matchValue;
            bestMatchLoc = matchLoc;
            bestAngle = angle;
            if (angle > 180) {
                bestAngle -= 360;
            }
        }
    }

    // 输出最佳匹配结果
    std::cout << "Best Match Location (x,y): " << bestMatchLoc.x << ", " << bestMatchLoc.y << endl;
    std::cout << "Best Match Value: " << bestMatchValue << endl;
    std::cout << "Best Angle: " << bestAngle << " degrees" << endl;

    // 可视化最佳匹配结果
    // cv::rectangle(img, bestMatchLoc, Point(bestMatchLoc.x + templ.cols, bestMatchLoc.y + templ.rows), Scalar(0, 255, 0), 2);
    cv::rectangle(img, bestMatchLoc, Point(bestMatchLoc.x + templ.cols, bestMatchLoc.y + templ.rows), Scalar(255, 255, 255), 2);
    cv::imshow("Best Match", img);
    cv::waitKey(0);

    return 0;
}

// 找到最佳匹配位置和角度
void findBestMatch(Mat &result, Point &bestMatchLoc, double &bestMatchValue) {
    double minVal, maxVal; // 最小值和最大值
    Point minLoc, maxLoc; // 最小值和最大值的位置

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    bestMatchLoc = maxLoc; // 最佳匹配位置
    bestMatchValue = maxVal; // 最佳匹配值
}
