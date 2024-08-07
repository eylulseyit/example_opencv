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

cv::Mat blendImages(cv::Mat& img2, cv::Mat& img1, cv::Mat& H) {
    // Warp the first image using the homography matrix
    cv::Mat img1_warped;
    cv::warpPerspective(img1, img1_warped, H, cv::Size(img1.cols + img2.cols, img1.rows));

    cv::imshow("1warped", img1_warped);

    // Create a mask for the warped image
    cv::Mat mask1 = cv::Mat::zeros(img1_warped.size(), CV_8UC1);
    cv::warpPerspective(cv::Mat::ones(img1.size(), CV_8UC1), mask1, H, cv::Size(img1.cols + img2.cols, img1.rows));

    // Copy the second image into the warped image space
    cv::Mat mask2 = cv::Mat::ones(img2.size(), CV_8UC1);
    cv::Mat img2_warped = cv::Mat::zeros(img1_warped.size(), img1_warped.type());
    img2.copyTo(img2_warped(cv::Rect(0, 0, img2.cols, img2.rows)));
    mask2.copyTo(mask1(cv::Rect(0, 0, mask2.cols, mask2.rows)));

    // Blend the images
    cv::Mat result = cv::Mat::zeros(img1_warped.size(), img1_warped.type());
    for (int y = 0; y < img1_warped.rows; y++) {
        for (int x = 0; x < img1_warped.cols; x++) {
            if (mask1.at<uchar>(y, x) == 0) {
                result.at<cv::Vec3b>(y, x) = img2_warped.at<cv::Vec3b>(y, x);
            } else if (mask1.at<uchar>(y, x) == 1) {
                result.at<cv::Vec3b>(y, x) = img1_warped.at<cv::Vec3b>(y, x);
            } else {
                cv::Vec3b color1 = img1_warped.at<cv::Vec3b>(y, x);
                cv::Vec3b color2 = img2_warped.at<cv::Vec3b>(y, x);
                result.at<cv::Vec3b>(y, x) = (color1 + color2) / 2;
            }
        }
    }

    return result;
}

int stitchDiff(cv::Mat img1, cv::Mat img2){

    if(img1.cols >1000 || img1.rows > 1000){
        cv::Mat resizedImage;
        cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));
        cv::resize(img2, img2, cv::Size(img2.cols / 2, img2.rows / 2));
    }
    // Detect ORB keypoints and descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);


    // Convert descriptors to CV_32F
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);

    // Use FLANN based matcher
    cv::BFMatcher matcher(cv::NORM_L2, true); // Using NORM_L2 and crossCheck set to true
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);//RANSAC

    std::cout<< matches.data();

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
    return a.distance < b.distance;
    });

    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < matches.size(); i++) {
        // Get the keypoints from the good matches
        obj.push_back(keypoints1[matches[i].queryIdx].pt);
        scene.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);
    // Blend the images

    cv::Mat img1_warped;
    cv::warpPerspective(img1, img1_warped, H, cv::Size(img1.cols + img2.cols, img1.rows));

    cv::imshow("1warped", img1_warped);

    // Create a mask for the warped image
    cv::Mat mask1 = cv::Mat::zeros(img1_warped.size(), CV_8UC1);
    cv::warpPerspective(cv::Mat::ones(img1.size(), CV_8UC1), mask1, H, cv::Size(img1.cols + img2.cols, img1.rows));
    cv::imshow("mask1", mask1);


    // Copy the second image into the warped image space
    cv::Mat mask2 = cv::Mat::ones(img2.size(), CV_8UC1);
    cv::Mat img2_warped = cv::Mat::zeros(img1_warped.size(), img1_warped.type());
    img2.copyTo(img2_warped(cv::Rect(0, 0, img2.cols, img2.rows)));
    mask2.copyTo(mask1(cv::Rect(0, 0, mask2.cols, mask2.rows)));
    cv::imshow("mask2",mask2);
    cv::imshow("img2_warped",img2_warped);
    cv::waitKey(0);
    return 0;
}

cv::Mat findHom(cv::Mat img1, cv::Mat img2){
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    /*std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);*/
    //key points detection and descriptions are found in lastFramekeypoints* , lastFrameDescriptors* respectively.

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);


    cv::Mat img_keypoints1;
    cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::Mat img_keypoints2;
    cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);//buraya kadar ok

    cv::imshow("Keypoints", img_keypoints1);
    cv::waitKey(2000);
    cv::imshow("Keypoints", img_keypoints2);
    cv::waitKey(2000);

    cv::BFMatcher matcher(cv::NORM_L2, true); // Using NORM_L2 and crossCheck set to true
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);//RANSAC

    std::cout<< matches.data();

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
    return a.distance < b.distance;
    });

    //sort(matches.begin(), matches.end());
    //const int numGoodMatches = matches.size() * 0.50;
    //matches.erase(matches.begin() + numGoodMatches, matches.end());
    cv::Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    //resize(imgMatches, imgMatches, cv::Size(), 0.1, 0.1);
    imshow("matches.jpg", imgMatches);
    cv::waitKey(2000);

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    cv::Mat H = findHomography(points2, points1, cv::RANSAC);
    return H;
}

void alphaBlend(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& outImage)
{
     // Find number of pixels.
     int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();
 
     // Get floating point pointers to the data matrices
     float* fptr = reinterpret_cast<float*>(foreground.data);
     float* bptr = reinterpret_cast<float*>(background.data);
     float* aptr = reinterpret_cast<float*>(alpha.data);
     float* outImagePtr = reinterpret_cast<float*>(outImage.data);
 
     // Loop over all pixesl ONCE
     for(
       int i = 0;
       i < numberOfPixels;
       i++, outImagePtr++, fptr++, aptr++, bptr++
     )
     {
         *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
     }
}


void stitch(cv::Mat img1, cv::Mat img2 )
{

    if(img1.cols >1000 || img1.rows > 1000){
        cv::Mat resizedImage;
        cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));
        cv::resize(img2, img2, cv::Size(img2.cols / 2, img2.rows / 2));
    }
    cv::Mat H = findHom(img1,img2);
    // Warp the second image to the first
    cv::Mat img2Warped;
    warpPerspective(img2, img2Warped, H, cv::Size(img1.cols + img2.cols, img1.rows));
    cv::waitKey(100);

    //make a black rectangle and make img2warped's left black(for the size of img1)
    //cv::Mat black = cv::Mat::zeros(img1.rows,img1.cols, img2Warped.type());
    //black.copyTo(img2Warped(cv::Rect(0, 0, black.cols, black.rows)));

    //crop img2warped
    img2Warped = img2Warped(cv::Range(0,img1.rows), cv::Range(img1.cols,img2Warped.cols));
    

    std::cout << "img1 Size: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "img2Warped Size: " << img2Warped.cols << "x" << img2Warped.rows << std::endl;

    cv::imshow("img1", img1);
    cv::imshow("img2Warped", img2Warped);
    cv::waitKey(1000);

    
    // Create a result image to hold the stitched image
    cv::Mat result = cv::Mat::zeros(img1.rows, img2Warped.cols +img1.cols, img1.type()); //change it when change others(copyto kullandıgında)
    // Copy the first image into the result image
    std::cout << "result Size: " << result.cols << "x" << result.rows << std::endl;
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2Warped.copyTo(result(cv::Rect(img1.cols, 0, img2Warped.cols, img2Warped.rows)));
    //float alpha = 0.5;
    //addWeighted(result, alpha, img2Warped, 1-alpha, 0, result);
    //result = reverseAddWeighted(result);
    //blending!!
    // Blend overlapping regions
    cv::imshow("Stitched Image", result);

    imwrite("stitchedBF.jpg", result);
    cv::waitKey(0);
}

int stitching2(cv::Mat img1, cv::Mat img2){
    cv::Mat pano;
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
     
    cv::waitKey(0);
    return 0;
}

int videoOp(){
    cv::VideoCapture capLeft("../videos/Hill1.mp4");
    cv::VideoCapture capRight("../videos/Hill2.mp4");
    if(!capLeft.isOpened() | !capRight.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
    return -1;
    }

    return 0;

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

    stitch(img1, img2);
    //stitchDiff(img1,img2);
    //videoOp();
    

    return 0;
}
