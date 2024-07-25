//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char** argv)
{

    int checkerBoard[2] = {12, 8};
    int fieldSize = 3.0f;//every grid's real life size is 3cm (for each square)
    // float fieldSize = 3.0f;
    std::vector<cv::String> fileNames;
    cv::glob("../leftcamera/Im_L_*.png", fileNames, false);//for fill the fileNames vector with all files in samples
    cv::Size patternSize(11, 7);
    std::vector<std::vector<cv::Point2f>> imgPoints(fileNames.size());//q
    std::vector<std::vector<cv::Point3f>> objPoints;//Q  
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    std::vector<cv::Point3f> objPoint;
    for (size_t i = 0; i < checkerBoard[0]-1; i++)//irl coordinates for 3D points
    {
        for (size_t j = 0; j < checkerBoard[1]-1; j++)
        {
            objPoint.push_back(cv::Point3f(i* fieldSize, j * fieldSize, 0));
        }
    }

    std::cout << "objpoint's size: " +objPoint.size() << std::endl;

    std::vector<cv::Point2f> imgPoint;
    std::size_t i = 0;

    for(auto const &f : fileNames)//read image files
    {
        std::cout << std::string(f) << std::endl;
        cv::Mat img = cv::imread(fileNames[i]);

        if (img.empty()) {
            std::cout << "Error: Unable to open image " << f << std::endl;
            continue;
        }
        cv::Mat gray;


        std::vector<cv::Point2f> corners;

        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        // Improve image contrast
        cv::equalizeHist(gray, gray);

        // Additional preprocessing (optional)
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        bool patternFound = cv::findChessboardCorners(gray,patternSize,imgPoints[i],cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE );
        if(patternFound){//cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1)
            //cornerSubPix(gray, corners[i], Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)); //???????????????
            cv::cornerSubPix(gray, imgPoints[i], cv::Size(11, 11), cv::Size(-1, -1), criteria);
            objPoints.push_back(objPoint);
        }else{
            std::cout << "Checkerboard not found in image: " << f << std::endl;
        }

        cv::drawChessboardCorners(img, patternSize, imgPoints[i], patternFound);
        cv::imshow("chessboard detected", img);
        cv::waitKey(100);
        i++;
    }

    std::cout << objPoints.size() << std::endl;
    std::cout << imgPoint.size() << std::endl;

    //up part works
    

    // Calibration

    cv::Matx33f cameraMatrix(cv::Matx33f::eye());//make the camera matrix a Identity matrix as default K
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Vec<float, 5> distCoeffs(0,0,0,0,0);//k
    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST +cv::CALIB_FIX_PRINCIPAL_POINT;
    cv::Size frameSize(1440, 1080);

    float error = cv::calibrateCamera(objPoints, imgPoints, frameSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);///RO? finds camera matrix
    
    std::cout << "Reprojection error: " << error << "\n cameraMatrix = \n"
              << cameraMatrix << "\n k = \n"
              << distCoeffs << std::endl;

    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33f::eye(),cameraMatrix, frameSize, CV_32FC1, mapX, mapY);

    for (auto const &f : fileNames)
    {
        std::cout << std::string(f) << std::endl;
        cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
        cv::Mat imgUndist;
        cv::remap(img, imgUndist, mapX, mapY, cv::INTER_LINEAR);

        cv::imshow("undistorted image", imgUndist);
        cv::waitKey(500);
    }


    return 0;
}