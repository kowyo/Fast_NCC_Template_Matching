#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load image and template
    Mat img = imread("../images/IMAGEB1.bmp", IMREAD_GRAYSCALE);
    Mat templ = imread("../images/pattern.bmp", IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        std::cerr << "Could not read the image or template file" << std::endl;
        return -1;
    }

    // // show the original image and template
    // cout << "Original image and template, press any key to continue" << endl;
    // imshow("Image", img);
    // imshow("Template", templ);
    // waitKey(0);

    // Define number of pyramid layers and rotation angles
    const int num_layers = 4;
    const int num_angles = 360; // Number of rotation angles to consider

    // Create image and template pyramids
    std::vector<Mat> img_pyramid(num_layers);
    std::vector<Mat> templ_pyramid(num_layers);
    img_pyramid[0] = img.clone();
    templ_pyramid[0] = templ.clone();
    for (int i = 1; i < num_layers; ++i) {
        pyrDown(img_pyramid[i - 1], img_pyramid[i]);
        pyrDown(templ_pyramid[i - 1], templ_pyramid[i]);
    }

    // Define NCC match result
    std::vector<Point> best_matches(num_layers);
    double best_score = -1.0;

    // Perform template matching at each rotation angle
    for (int angle = 0; angle < num_angles; ++angle) {
        Mat rotated_templ;
        Point center(templ.cols / 2, templ.rows / 2);
        Mat rotation = getRotationMatrix2D(center, angle * (360.0 / num_angles), 1.0);
        warpAffine(templ, rotated_templ, rotation, templ.size());

        // show the rotated template
        cout << "Rotated template, press any key to continue" << endl;
        imshow("Rotated template", rotated_templ);
        waitKey(0);

        // Perform matching on each layer of the pyramid
        for (int i = num_layers - 1; i >= 0; --i) {
            Mat result;
            matchTemplate(img_pyramid[i], rotated_templ, result, TM_CCORR_NORMED);

            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal > best_score) {
                best_score = maxVal;
                best_matches[i] = maxLoc * pow(2, i);
            }
        }
    }

    // Draw the best match on the original image
    for (int i = 0; i < num_layers; ++i) {
        circle(img, best_matches[i], 5, Scalar(255, 0, 0), 2);
    }

    // Display the result
    imshow("Matching result", img);
    waitKey(0);

    return 0;
}
