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

    /*cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);*/



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

int blend(cv::Mat img1, cv::Mat img2){

    cv::Mat g1 = img1.clone();
    cv::Mat g2 = img2.clone();

    std::vector<cv::Mat> gPyr1 = {g1};
    std::vector<cv::Mat> gPyr2 = {g2};

    for (int i = 0; i < 6; i++) {
        cv::pyrDown(g1, g1);
        gPyr1.push_back(g1);
        cv::pyrDown(g2, g2);
        gPyr2.push_back(g2);
    }

    std::vector<cv::Mat> lp1= {gPyr1[5]};
    std::vector<cv::Mat> lp2= {gPyr2[5]};
    for (int i = 5; i > 0; i--) {
        cv::Mat GE1, L1;
        cv::pyrUp(gPyr1[i], GE1, gPyr1[i-1].size());
        cv::subtract(gPyr1[i-1], GE1, L1);
        lp1.push_back(L1);

        cv::Mat GE2, L2;
        cv::pyrUp(gPyr2[i], GE2, gPyr2[i-1].size());
        cv::subtract(gPyr2[i-1], GE2, L2);
        lp2.push_back(L2);
    }

    // Now add left and right halves of images in each level
    std::vector<cv::Mat> LS;
    for (size_t i = 0; i < lp1.size(); i++) {
        cv::Mat la = lp1[i];
        cv::Mat lb = lp2[i];
        cv::Mat ls;
        cv::hconcat(la(cv::Rect(0, 0, la.cols / 2, la.rows)), 
                    lb(cv::Rect(lb.cols / 2, 0, lb.cols / 2, lb.rows)), ls);
        LS.push_back(ls);
    }

    // Now reconstruct
    cv::Mat ls_ = LS[0];
    for (size_t i = 1; i < LS.size(); i++) {
        cv::pyrUp(ls_, ls_, LS[i].size());
        cv::add(ls_, LS[i], ls_);
    }

    // Image with direct connecting each half
    cv::Mat real;
    cv::hconcat(img1(cv::Rect(0, 0, img1.cols / 2, img1.rows)), 
                img2(cv::Rect(img2.cols / 2, 0, img2.cols / 2, img2.rows)), real);

    // Save results

    cv::imshow("Pyramid_blending2.jpg", ls_);
    cv::imshow("Direct_blending.jpg", real);
    cv::imwrite("Pyramid_blending2.jpg", ls_);
    cv::imwrite("Direct_blending.jpg", real);

    return 0;

}

void findCoord(int& xR, int xL, int sizeL, int sizeR){
    int diff= sizeL+sizeR -(sizeL*2);
    std::cout << "diff: " << diff << std::endl;
    xR = xL-(sizeL -diff);
}

void counter(int& ctr, float& alpha){
    
    if(ctr ==10){
        ctr = 1;
        alpha = 0.1;
    }
    else{
        ctr++;
    }
}

void featherblend(cv::Mat& leftImg, cv::Mat rightImg){
    int right_width = 20;
    cv::Rect roi(leftImg.cols - right_width, 0, right_width, leftImg.rows);
    cv::Mat submat = leftImg(roi);

    /*cv::imshow("not warped",rightImgNotWarped);
    cv::imshow("warped",rightImg);*/

    cv::MatIterator_<cv::Vec3b> it; // = src_it.begin<cv::Vec3b>();
    int ctr = 0;
    float alpha = 0.1;//the percentage of blending
    int index;
    int xR;
    int curRow = -1;//this is for fixing the alpha values
    int count =0;
  
    //iterates the blending area(10 x pixels)
    for (it = submat.begin<cv::Vec3b>(); it != submat.end<cv::Vec3b>(); ++it)
    {
        index = it - submat.begin<cv::Vec3b>();// for the coordinate of the current point

        int y = index / submat.cols;  // Row coordinate of both images
        int x = index % submat.cols;  // Column coordinate of left image

        //findCoord(xR, xL,leftImg.rows, rightImg.rows);

        
        if(x != curRow){
            curRow = x;
            counter(ctr, alpha);
        }
        alpha *=ctr;
        count++;

        if(count == 23){
            std::cout << "current count: " << count << std::endl;
            std::cout << "left's row: " << curRow << std::endl;
            std::cout << "right's x: " << x << std::endl;

        
            std::cout << "alpha: " << alpha << std::endl;
            break;
        }

        cv::Vec3b pixel =rightImg.at<cv::Vec3b>(y, x);
        if(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0){
            continue;
        }
        (*it)[0] = static_cast<uchar>((1.0-alpha) * (*it)[0] + (alpha) * pixel[0]);
        (*it)[1] = static_cast<uchar>((1.0-alpha) * (*it)[1] + (alpha) * pixel[1]);
        (*it)[2] = static_cast<uchar>((1.0-alpha) * (*it)[2] + (alpha) * pixel[2]);
        /*float perc = alpha* 255;

        *it = (perc/255 *pixel ) + ((255 - perc)/255 * (*it));*/

        /*(*it)[0] = static_cast<uchar>( ( ( (alpha*255) /255)  * ((*it)[0] )  )      +     ((alpha/255) * pixel[0]));
        (*it)[1] = static_cast<uchar>( ( ( (1.0-alpha) /255)  * ((*it)[0] )  )      +     ((alpha/255) * pixel[0]));
        (*it)[2] = static_cast<uchar>( ( ( (1.0-alpha) /255)  * ((*it)[0] )  )      +     ((alpha/255) * pixel[0]));*/
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
    cv:: Mat img2Warpedf = img2Warped.clone();
    featherblend(img1, img2Warped);
    img2Warped = img2Warped(cv::Range(0,img1.rows-10), cv::Range(img1.cols,img2Warped.cols));//extra 10 pixels for blending
    

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

    //blend(img1, img2Warped);

    //float alpha = 0.5;
    //addWeighted(result, alpha, img2Warped, 1-alpha, 0, result);
    //result = reverseAddWeighted(result);
    //blending!!
    // Blend overlapping regions
    cv::imshow("Stitched Image", result);

    imwrite("stitchedBF.jpg", result);


    cv::waitKey(0);
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

    stitch(img1, img2);
    //stitchDiff(img1,img2);
    //videoOp();
    

    return 0;
}
