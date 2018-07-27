// make方法
// g++ -std=c++14 -lm -I/usr/include/eigen3 -o program ./*.cpp `pkg-config opencv --cflags --libs`

// c++14 standard liblary
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>

// OpenCV (ver 3.1)
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

// Eigen
#include <Eigen/Eigen>
#include <Eigen/Core>

using namespace std;
using namespace cv;

// コマンドライン引数
enum ARG{
    PROG_NAME,          // プログラム名
    SRC_IMG,            // 元画像
    TRANS_IMG,          // 変換後の画像
    NUM_OF_NEED_ARG     // コマンドライン引数の個数
};

int main(int argc, char* argv[])
try{
    Mat     img[2];         // 入力画像、変換後の画像
    Mat     gray[2];        // imgそれぞれをグレースケール化した画像
    Mat     drawKpts[2];    // imgそれぞれに特徴点検出結果を描画した画像
    string  filepath[2];    // 入力画像、変換後の画像のファイルパス

    // コマンドライン引数が足りない
    if(argc < NUM_OF_NEED_ARG){
        throw("not enough num of arguments");
    }

    

    // ２つの画像のファイルパスを取得する
    filepath[0] = argv[SRC_IMG];
    filepath[1] = argv[TRANS_IMG];

    // 画像を取得する
    img[0] = imread(filepath[0], -1);
    img[1] = imread(filepath[1], -1);

    imshow("input",  img[0]);
    imshow("output", img[1]);
    waitKey();

    Mat img2[2];

    img2[0] = Mat(img[0].rows, img[0].cols,  CV_8UC3);
    img2[1] = Mat(img[1].rows, img[1].cols, CV_8UC3);

    cerr << "img[0]  : rows = " << img[0].rows << " cols = " << img[0].cols      << " channel =  " << img[0].channels() << endl;;
    cerr << "img[1]  : rows = " << img[1].rows << " cols = " << img[1].cols      << " channel =  " << img[1].channels() << endl;
    cerr << "img2[0] : rows = " << img2[0].rows << " cols = " << img2[0].cols   << " channel =  " << img2[0].channels() << endl;
    cerr << "img2[1] : rows = " << img2[1].rows << " cols = " << img2[1].cols   << " channel =  " << img2[1].channels() << endl;
 
    for(int x = 0; x < img[0].cols; x++){
        for(int y = 0; y < img[0].rows; y++){
            
            int alpha[2];
            
            alpha[0] = img[0].at<Vec4b>(y, x)[3];
            alpha[1] = img[1].at<Vec4b>(y, x)[3];

            // 背景色は全て白として扱う
            if(alpha[0] == (uchar)0x00 ){
                img2[0].at<Vec3b>(y, x)[0] = 0x00;
                img2[0].at<Vec3b>(y, x)[1] = 0x00;
                img2[0].at<Vec3b>(y, x)[2] = 0x00;
            }
            else{
                img2[0].at<Vec3b>(y, x)[0] = img[0].at<Vec4b>(y, x)[0];
                img2[0].at<Vec3b>(y, x)[1] = img[0].at<Vec4b>(y, x)[1];
                img2[0].at<Vec3b>(y, x)[2] = img[0].at<Vec4b>(y, x)[2];
            }

            // 背景色は全て白として扱う
            if( alpha[1] == (uchar)0x00){
                img2[1].at<Vec3b>(y, x)[0] = 0x00;
                img2[1].at<Vec3b>(y, x)[1] = 0x00;
                img2[1].at<Vec3b>(y, x)[2] = 0x00;
            }
            else{
                img2[1].at<Vec3b>(y, x)[0] = img[1].at<Vec4b>(y, x)[0];
                img2[1].at<Vec3b>(y, x)[1] = img[1].at<Vec4b>(y, x)[1];
                img2[1].at<Vec3b>(y, x)[2] = img[1].at<Vec4b>(y, x)[2];
            }


        }
    }

    cerr << "(int)(uchar)255 = " << (int)((uchar)255) << endl;
    cerr << "img[0].(" << 0 << ", " << 0 << ") = " << (int)(uchar)img[0].at<Vec4b>(0, 0)[0]   << "," << (int)(unsigned char)img[0].at<Vec4b>(0, 0)[1] << "," << (int)(unsigned char)img[0].at<Vec4b>(0, 0)[2] << "," << (int)(unsigned char)img[0].at<Vec4b>(0, 0)[3] << "," << endl;


    imshow("input(non alpha)",  img2[0]);
    imshow("output(non alpha)", img2[1]);
    waitKey();

    img[0].release();
    img[1].release();
    img[0] = img2[0].clone();
    img[1] = img2[1].clone();

    imshow("input", img[0]);
    imshow("output", img[1]);
    waitKey();

    // グレイ画像へ変換する
    cvtColor(img[0], gray[0], COLOR_BGR2GRAY);
    cvtColor(img[1], gray[1], COLOR_BGR2GRAY);

    //gray[0] = img[0].clone();
    //gray[1] = img[1].clone();

    imshow("gray", gray[0]);
    imshow("gray", gray[1]);
    waitKey();

    // 特徴量記述に用いるアルゴリズム
    auto algorithm = AKAZE::create();   // A-KAZEを用いる
    vector<KeyPoint> kpts[2];           // 特徴点
    Mat desc[2];                        // 特徴量記述

    // 特徴点の検出を行う
    algorithm->detect(img[0], kpts[0]);
    algorithm->detect(img[1], kpts[1]);

    // 特徴量の記述を行う
    algorithm->compute(img[0], kpts[0], desc[0]);
    algorithm->compute(img[1], kpts[1], desc[1]);

    // マッチングを行うためのオブジェクト
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");   // マッチャーではBruteForceを使用する
    vector<DMatch> crossMatch, match12, match21;                                // クロスマッチ結果, マッチ（入力画像→変換後の画像，変換後の画像→入力画像）

    // マッチングを行う
    matcher->match(desc[0], desc[1], match12);
    matcher->match(desc[1], desc[0], match21);

    // クロスマッチングを行う
    // 相互に存在する特徴点のみを残して精度を高める
    for(auto i = 0; i < match12.size(); i++){
        DMatch forward = match12[i];
        DMatch backward = match21[forward.trainIdx];
        if (backward.trainIdx == forward.queryIdx)
        {
            crossMatch.push_back(forward);
            //circle(img[1], cv::Point(kpts[1][forward.trainIdx].pt.x, kpts[1][forward.trainIdx].pt.y), 10, Scalar(0,0,0), -1, 4);
        }
    }

    Mat dest;
    drawMatches(img[0], kpts[0], img[1], kpts[1], crossMatch, dest);
    imshow("test", dest);
    waitKey();
 
    double s, s2;   // スケールs,s^2

    // 2行n列 X, X'
    Mat X(2, crossMatch.size(), CV_64FC1);
    Mat Xd(2, crossMatch.size(), CV_64FC1);
    //Mat K(2, crossMatch.size(), CV_64FC1);
    Mat K(2, 2, CV_64FC1);

    double xxt = 0;
    double xdxdt = 0;


    for(int i = 0; i < crossMatch.size(); i++){
        DMatch& dm = crossMatch[i];
        
        Point2d pt[2];
        pt[0].x = kpts[0][dm.queryIdx].pt.x - img[0].cols / 2.0f;
        pt[0].y = kpts[0][dm.queryIdx].pt.y - img[0].rows / 2.0f;
        pt[1].x = kpts[1][dm.queryIdx].pt.x - img[1].cols / 2.0f;   
        pt[1].y = kpts[1][dm.queryIdx].pt.y - img[1].rows / 2.0f;

        // xixi^tを求める
        xxt += 
            pt[0].x * pt[0].x + pt[0].y * pt[0].y;
            //kpts[0][dm.queryIdx].pt.x * kpts[0][dm.queryIdx].pt.x +
            //kpts[0][dm.queryIdx].pt.y * kpts[0][dm.queryIdx].pt.y;

        // xi'xi'^tを求める
        xdxdt += 
            pt[1].x * pt[1].x + pt[1].y * pt[1].y;
            //kpts[1][dm.trainIdx].pt.x * kpts[1][dm.trainIdx].pt.x +
            //kpts[1][dm.trainIdx].pt.y * kpts[1][dm.trainIdx].pt.y;

        // 行列Xを求める
        X.at<double>(0, i) = pt[0].x;
        X.at<double>(1, i) = pt[0].y;
        //X.at<double>(0, i) = kpts[0][dm.queryIdx].pt.x;
        //X.at<double>(1, i) = kpts[0][dm.queryIdx].pt.y;

        // 行列X'を求める
        Xd.at<double>(0, i) = pt[1].x;
        Xd.at<double>(1, i) = pt[1].y;
//        Xd.at<double>(0, i) = kpts[1][dm.trainIdx].pt.x;
//        Xd.at<double>(1, i) = kpts[1][dm.trainIdx].pt.y;

        // 座標を元の画面の中心を原点とした座標系へ変換する
 

    }



    // s^2 及び s を求める
    s2 = xdxdt / xxt;
    //s2 = xdxdt*xdxdt / (xxt*xxt);
    s = sqrt(s2);

    cerr << "scale = " << s << endl;

    //X /= s;
    //K = (1.0f / s) * Xd * (X.t() * X ).inv() * X.t();
    K = (1.0f / s) * Xd * (X.t() * X ).inv() * X.t();

    cerr << "K:" << endl;
    cerr << K << endl;

    cerr << "|K| = " <<  determinant(K) << endl;
    K /= determinant(K);

    cerr << "K/|K|:" << endl;
    cerr << K << endl;

    waitKey(0);

    auto RAD_TO_DEG = [](double rad){return rad*180.0f/CV_PI;};
    cout << "K(0,0) : acosθ = " << RAD_TO_DEG(acos(K.at<double>(0,0))) << endl;
    cout << "K(1,1) : acosθ = " << RAD_TO_DEG(acos(K.at<double>(1,1))) << endl;
    cout << "K(0,1) : asinθ = " << RAD_TO_DEG(acos(K.at<double>(0,1))) << endl;
    cout << "K(1,0) : asinθ = " << RAD_TO_DEG(acos(K.at<double>(1,0))) << endl;

    return 0;
}
catch(...){
    return 1;
}
