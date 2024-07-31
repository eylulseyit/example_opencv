#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <opencv4/opencv2/opencv.hpp>

std::vector<cv::Point3f> createObjectPoints(int width, int height, float squareSize) {
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            objectPoints.emplace_back(j * squareSize, i * squareSize, 0.0f);
        }
    }
    return objectPoints;
}

int main(int argc, char** argv){
    int checkerBoard[2] = {9, 7}; // Number of internal corners per a chessboard row and column
    float fieldSize = 3.25f; // Real life size of each square in cm

    std::vector<cv::String> fileNames;
    cv::glob("../imgexamples/left0*.jpg", fileNames, false); // Adjust the path as necessary

    cv::Size patternSize(checkerBoard[0] - 1, checkerBoard[1] - 1);
    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<std::vector<cv::Point3f>> objPoints;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    // Preparing the object points
    std::vector<cv::Point3f> objPoint = createObjectPoints(checkerBoard[0]-1, checkerBoard[1]-1, fieldSize);
    
    for (const auto& f : fileNames) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) {
            std::cerr << "Error: Unable to open image " << f << std::endl;
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        // Additional preprocessing 
        cv::equalizeHist(gray, gray);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        std::vector<cv::Point2f> corners;
        bool patternFound = cv::findChessboardCorners(gray, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (patternFound) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            imgPoints.push_back(corners);
            objPoints.push_back(objPoint);
        } else {
            std::cout << "Checkerboard not found in image: " << f << std::endl;
        }

        cv::drawChessboardCorners(img, patternSize, corners, patternFound);
        cv::imshow("chessboard detected", img);
        cv::waitKey(100);
    }

    std::cout << "Number of valid images: " << imgPoints.size() << std::endl;

    // Calibration
    cv::Matx33f cameraMatrix = cv::Matx33f::eye();
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Vec<float, 5> distCoeffs(0, 0, 0, 0, 0);
    int flags = cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT;

    if (imgPoints.size() > 0) {
        cv::Size frameSize = cv::imread(fileNames[0]).size();
        //cv:: Size& fs = frameSize*;
        double error = cv::calibrateCamera(objPoints, imgPoints, frameSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);
        //double error = cv::fisheye::calibrate(objPoints, imgPoints, frameSize, cameraMatrix, distCoeffs, rvecs,tvecs,flags, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, DBL_EPSILON));

        std::cout << "Reprojection error: " << error << "\ncameraMatrix = \n" << cameraMatrix << "\nk = \n" << distCoeffs << std::endl;

        // Reprojection error for each image
        double totalError = 0;
        for (size_t i = 0; i < objPoints.size(); ++i) {
            std::vector<cv::Point2f> reprojectedPoints;
            cv::projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, reprojectedPoints);
            double error = cv::norm(imgPoints[i], reprojectedPoints, cv::NORM_L2);
            totalError += error * error;
        }
        double meanError = std::sqrt(totalError / objPoints.size());
        std::cout << "Mean Reprojection Error: " << meanError << std::endl;

        // Undistort images
        cv::Mat mapX, mapY;
        cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33f::eye(), cameraMatrix, frameSize, CV_32FC1, mapX, mapY);

        int count = 1;//for download fixed files
        for (const auto& f : fileNames) {
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            cv::Mat imgUndist;
            cv::remap(img, imgUndist, mapX, mapY, cv::INTER_LINEAR);

            std::ostringstream output_path;
            output_path << "/home/eylul/example_opencv/fixedimages/a" << count << ".jpg";

            // Save the fixed image to the specified path
            bool result = cv::imwrite(output_path.str(), imgUndist);
            //path = f
            //cv::imwrite("/home/eylul/example_opencv/fixedimages\\frame%d.jpg" % count , imgUndist);

            cv::imshow("undistorted image", imgUndist);
            cv::waitKey(2000);
            count++;
        }
    } else {
        std::cerr << "Not enough valid images for calibration." << std::endl;
    }
    return 0;
}