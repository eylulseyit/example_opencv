#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <opencv4/opencv2/opencv.hpp>

#include "opencv2/stitching.hpp"


void stitching(cv::Mat img1, cv::Mat img2 ){
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);


    cv::Mat img_keypoints1;
    cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::Mat img_keypoints2;
    cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);//buraya kadar ok

    cv::imshow("Keypoints", img_keypoints1);
    cv::waitKey(2000);
    cv::imshow("Keypoints", img_keypoints2);
    cv::waitKey(2000);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);//RANSAC

    std::cout<< matches.data();

    sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * 0.15;
    matches.erase(matches.begin() + numGoodMatches, matches.end());
    cv::Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    resize(imgMatches, imgMatches, cv::Size(), 0.5, 0.5);
    imshow("matches.jpg", imgMatches);
    cv::waitKey(20000);

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    cv::Mat H = findHomography(points1, points2, cv::RANSAC);

    cv::Mat img2Warped;
    warpPerspective(img2, img2Warped, H, cv::Size(img1.cols + img2.cols, img1.rows));

    // Create a mask for blending
    cv::Mat mask1 = cv::Mat::ones(img1.size(), CV_8U) * 255;
    cv::Mat mask2 = cv::Mat::ones(img2.size(), CV_8U) * 255;
    warpPerspective(mask2, mask2, H, cv::Size(img1.cols + img2.cols, img1.rows));

    // Blend the images using multi-band blending
    cv::Mat result(img1.rows, img1.cols + img2.cols, img1.type(), cv::Scalar::all(0));
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2Warped.copyTo(result(cv::Rect(0, 0, img2Warped.cols, img2Warped.rows)), mask2);

    // Save and display the result
    imwrite("stitched.jpg", result);
    resize(result, result, cv::Size(), 0.5, 0.5);
    imshow("Stitched Image", result);
    cv::waitKey(0);




}

int main(int argc, char** argv)
{
    cv::Mat img1 = cv::imread("../stitching/left.jpg");

    if (img1.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat img2 = cv::imread("../stitching/right.jpg");
    if (img1.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    stitching(img1, img2);
    /*cv::Mat pano; CALISAN STITCHING
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
 
// Array for pictures
    std::vector<cv::Mat> imgs(2);
    imgs[0] = img1;
    imgs[1] = img2;

     
    // Create a Stitcher class object with mode panoroma
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
     
    // Command to stitch all the images present in the image array
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
 
    if (status != cv::Stitcher::OK)
    {
        // Check if images could not be stitched
        // status is OK if images are stitched successfully
        std::cout << "Can't stitch images\n";
        return -1;
    }
     
    // Store a new image stitched from the given 
    //set of images as "result.jpg"
    imwrite("result.jpg", pano);
     
    // Show the result
    imshow("Result", pano);
     
    cv::waitKey(0);*/

    return 0;
}
