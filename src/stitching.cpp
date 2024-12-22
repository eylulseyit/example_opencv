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

cv::Mat findHom(cv::Mat img1, cv::Mat img2){
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
    //key points detection and descriptions are found in lastFramekeypoints* , lastFrameDescriptors* respectively.
    
    //another keypoint detection function usage for testing
    /*cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);*/



    cv::Mat img_keypoints1;
    cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::Mat img_keypoints2;
    cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);//buraya kadar ok

    //seeing results for testing
    /*cv::imshow("Keypoints", img_keypoints1);
    cv::waitKey(2000);
    cv::imshow("Keypoints", img_keypoints2);
    cv::waitKey(2000);*/

    cv::BFMatcher matcher(cv::NORM_L2, true); // Using NORM_L2 and crossCheck set to true
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);//RANSAC

    std::cout<< matches.data();

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
    return a.distance < b.distance;
    });

    //optimizing the matchesPoints numbers, but later, I decided it doesn't matter
    //sort(matches.begin(), matches.end());
    //const int numGoodMatches = matches.size() * 0.50;
    //matches.erase(matches.begin() + numGoodMatches, matches.end());
    
    cv::Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    //resize(imgMatches, imgMatches, cv::Size(), 0.1, 0.1);
    /*imshow("matches.jpg", imgMatches);
    cv::waitKey(2000);*/

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    cv::Mat H = findHomography(points2, points1, cv::RANSAC);
    return H;
}

void featherblend(cv::Mat& leftImg, cv::Mat rightImg){

    /*cv::imshow("img1", leftImg);
    cv::imshow("img2", rightImg);*/
    int width = 100;
    cv::Rect roi(leftImg.cols - width, 0, width, leftImg.rows);
    cv::Mat submat = leftImg(roi);

    /*cv::imshow("not warped",rightImgNotWarped);
    cv::imshow("warped",rightImg);*/
    //cv::imshow("warped",submat);

    float alpha;//the percentage of blending
    int index;
    int curRow = -1;//this is for fixing the alpha values

    cv::Vec3b * currentRow;
    int a = leftImg.cols - width;
    float halfWidth = width/2;

    for (int j = 0; j < submat.rows; ++j)// for y axis
    {
        currentRow = submat.ptr<cv::Vec3b>(j);
        alpha = 1.0f/width;

        
        //std::cout << "a " << a<< std::endl;


        for (int i = 0; i < submat.cols; ++i)//for x axis
        {
            //std::cout << "alpha " << alpha<< std::endl;

            cv::Vec3b pixel = rightImg.at<cv::Vec3b>(j, i+ a);
            currentRow[i] = (alpha * pixel) + ((1.0f - alpha) * (currentRow[i]));

            // Ramp up alpha from 1/width to 1, then ramp down back to 1/width
                alpha = (i + 1) * (1.0f / width);  // Increment alpha up to 1
            
        }


    }
}


cv::Mat stitch(cv::Mat img1, cv::Mat img2, cv::Mat H )
{

    // Warp the second image to the first
    cv::Mat img2Warped;
    warpPerspective(img2, img2Warped, H, cv::Size(img1.cols + img2.cols, img1.rows));
    cv::waitKey(100);
    
    //crop img2warped
    cv:: Mat img2Warpedf = img2Warped.clone();
    featherblend(img1, img2Warped);
    img2Warped = img2Warped(cv::Range(0,img1.rows-10), cv::Range(img1.cols,img2Warped.cols));//extra 10 pixels for blending
    

    //std::cout << "img1 Size: " << img1.cols << "x" << img1.rows << std::endl;
    //std::cout << "img2Warped Size: " << img2Warped.cols << "x" << img2Warped.rows << std::endl;

    /*cv::imshow("img1", img1);
    cv::imshow("img2Warped", img2Warped);
    cv::waitKey(1000);*/
    
    // Create a result image to hold the stitched image
    cv::Mat result = cv::Mat::zeros(img1.rows, img2Warped.cols +img1.cols, img1.type()); //change it when change others(copyto kullandıgında)
    // Copy the first image into the result image
    //std::cout << "result Size: " << result.cols << "x" << result.rows << std::endl;
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));


    img2Warped.copyTo(result(cv::Rect(img1.cols, 0, img2Warped.cols, img2Warped.rows)));


    cv::imshow("Stitched Image", result);

    //imwrite("stitchedBF.jpg", result);


    cv::waitKey(0);
    return result;
}

int stitching2(cv::Mat img1, cv::Mat img2){// stitching module from opencv
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


int main(int argc, char** argv)
{
    cv::Mat img1 = cv::imread("../stitching/left3.jpg");

    if (img1.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat img2 = cv::imread("../stitching/right3.jpg");
    if (img1.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }  


    cv::Mat H = findHom(img1,img2);

    stitch(img1, img2, H);
    //videoOp();

    return 0;
}
