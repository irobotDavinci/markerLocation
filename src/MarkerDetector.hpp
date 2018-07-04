#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cstddef>
#include <opencv2/opencv.hpp>

#include "Marker.hpp"

using namespace cv;
using namespace std;


class MarkerDetector
{
public:
  typedef vector<Point> PointsVector;
  typedef vector<PointsVector> ContoursVector;

  MarkerDetector(float markerSize3d);

  bool processFrame(const Mat& frame,Mat_<float>& camMatrix,Mat_<float>& distCoeff,vector<Marker>& markers);


public:
  void prepareImage(const Mat& src,Mat& grayscale);
  void performThreshold(const Mat& grayscale,Mat& thresholdImg);
  void findContour(cv::Mat& thresholdImg, ContoursVector& contours, int minContourPointsAllowed) const;
  void findCandidates(const ContoursVector& contours,vector<Marker>& detectedMarkers);

  bool recognizeMarkers(const Mat& grayscale,vector<Marker>& detectedMarkers);
  void estimatePosition(vector<Marker>& detectedMarkers,Mat_<float>& camMatrix,Mat_<float>& distCoeff);
  bool findMarkers(const Mat& frame,vector<Marker>& detectedMarkers);

public:

  Size markerSize2d;

  ContoursVector m_contours;
  vector<Point3f> m_markerCorners3d;
  vector<Point2f> m_markerCorners2d;

  Mat m_grayscaleImage;
  Mat m_thresholdImg;
  Mat canonicalMarkerImage;

  //vector<Transformation> m_transformations;

  float m_minContourLengthAllowed;
};

struct BGRAVideoFrame
{
  size_t width;
  size_t height;
  size_t stride;

  unsigned char * data;
};

template <typename T>
string ToString(const T& value)
{
    ostringstream stream;
    stream << value;
    return stream.str();
}



