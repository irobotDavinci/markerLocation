#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<sstream>       //stringstream

#include "MarkerDetector.hpp"

using namespace cv;
using namespace std;

float perimeter(const vector<Point2f> &a)//求多边形周长。
{
  float sum=0,dx,dy;
  for(size_t i=0;i<a.size();i++)
    {
      size_t i2=(i+1) % a.size();

      dx = a[i].x - a[i2].x;
      dy = a[i].y - a[i2].y;

      sum += sqrt(dx*dx + dy*dy);
    }

  return sum;
}

MarkerDetector::MarkerDetector(float markerSize3d)
 {
    m_minContourLengthAllowed = 100.0;
    markerSize2d = Size(100,100);

   bool centerOrigin = true;
   if(centerOrigin)
     {
       //grid:56mm or 157mm
       float w = markerSize3d;
       m_markerCorners3d.push_back(Point3f(-w/2,-w/2,0));
       m_markerCorners3d.push_back(Point3f(+w/2,-w/2,0));
       m_markerCorners3d.push_back(Point3f(+w/2,+w/2,0));
       m_markerCorners3d.push_back(Point3f(-w/2,+w/2,0));
     }
   else
     {
       m_markerCorners3d.push_back(Point3f(0,0,0));
       m_markerCorners3d.push_back(Point3f(1,0,0));
       m_markerCorners3d.push_back(Point3f(1,1,0));
       m_markerCorners3d.push_back(Point3f(0,1,0));
     }

   m_markerCorners2d.push_back(Point2f(0,0));
   m_markerCorners2d.push_back(Point2f(markerSize2d.width-1,0));
   m_markerCorners2d.push_back(Point2f(markerSize2d.width-1,markerSize2d.height-1));
   m_markerCorners2d.push_back(Point2f(0,markerSize2d.height-1));

}

void MarkerDetector::prepareImage(const Mat& src,Mat& grayscale)
{
  //彩色转换成灰色图像
  if(1 == src.channels()) {
    grayscale = src;
    return;
  }
  cvtColor(src,grayscale,CV_BGRA2GRAY);
}

//绝对阈值结果取决于光照条件和软强度变化。采用自适应阈值法，以像素为单位，将给定半径内的所有像素的平均强度作为该像素的强度，使接下来的轮廓检测更具有鲁棒性。
void MarkerDetector::performThreshold(const Mat& grayscale,Mat& thresholdImg)
{
  /*输入图像  
  //输出图像  
  //使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值  
  //自适应阈值算法使用：CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C   
  //取阈值类型：必须是下者之一  
  //CV_THRESH_BINARY,  
  //CV_THRESH_BINARY_INV  
  //用来计算阈值的象素邻域大小: 3, 5, 7, ...  
  */  
  adaptiveThreshold(grayscale,//Input Image
                    thresholdImg,//Result binary image
                    255,
                    ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY_INV,
                    7,
                    7
                    );
#ifdef SHOW_DEBUG_IMAGES
  imshow("Threshold image",thresholdImg);
#endif
}

//检测所输入的二值图像的轮廓，返回一个多边形列表，其每个多边形标识一个轮廓，小轮廓不关注，不包括标记...
void MarkerDetector::findContour(cv::Mat& thresholdImg, ContoursVector& contours, int minContourPointsAllowed) const
{
  ContoursVector allContours;
  findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  contours.clear();
  for(size_t i=0;i<allContours.size();i++) {
    int contourSize = allContours[i].size();
    //if(contourSize > minContourPointsAllowed) {
    if(contourSize > 1) {
      contours.push_back(allContours[i]);
    }
  }
  //Mat result(src.size(),CV_8U,Scalar(0)); 
  //drawContours(result,detectedMarkers,-1,Scalar(255),2); 
  //imshow("AR based marker...",result);
#ifdef SHOW_DEBUG_IMAGES
  {
    Mat contoursImage(thresholdImg.size(), CV_8UC1);
    contoursImage = Scalar(0);
    drawContours(contoursImage, contours, -1, cv::Scalar(255), 2, CV_AA);
    imshow("Contours",contoursImage);

    Mat allcontoursImage(thresholdImg.size(), CV_8UC1);
    allcontoursImage = Scalar(0);
    drawContours(allcontoursImage, allContours, -1, cv::Scalar(255), 2, CV_AA);
    imshow("AllContours",allcontoursImage);
  }   
#endif
}

// 1.检测多边形，通过判断多边形定点数量是否为4，四边形是否足够大，过滤非候选区域。
// 2.根据候选区域之间距离进一步筛选，得到最终的候选区域，并使得候选区域的顶点坐标逆时针排列。
void MarkerDetector::findCandidates(const ContoursVector& contours,vector<Marker>& detectedMarkers)
{
  vector<Point> approxCurve;
  vector<Marker> possibleMarkers;

  //For each contour,分析它是不是像marker，找到候选者
  for(size_t i=0;i<contours.size();i++)
    {
      // 求出与轮廓近似的多边形
      double eps = contours[i].size()*0.05;
      approxPolyDP(contours[i],approxCurve,eps,true);

      // marker只有四个顶点
      if(approxCurve.size() != 4)
        continue;

      // 检查轮廓是否是凸边形
      if(!isContourConvex(approxCurve))
        continue;

      // 求当前四边形各顶点之间的最短距离,确保连续点之间的距离足够大
      float minDist = 1e10;
      for(int i=0;i<4;i++)
        {
          Point side = approxCurve[i] - approxCurve[(i+1)%4];
          float squaredSideLength = side.dot(side);
          minDist = min(minDist,squaredSideLength);
        }

      //距离小的话就退出本次循环
      if(minDist<m_minContourLengthAllowed)
        continue;

      //所有的测试通过了，保存标识候选，当四边形大小合适，则将该四边形maker放入possibleMarkers容器内
      Marker m;
      for(int i=0;i<4;i++)
        m.points.push_back(Point2f(approxCurve[i].x,approxCurve[i].y));

      // 逆时针保存这些点
      Point v1 = m.points[1] - m.points[0];
      Point v2 = m.points[2] - m.points[0];
      double o = (v1.x * v2.y) - (v1.y * v2.x);
      if(o<0.0) 
        swap(m.points[1],m.points[3]);

      possibleMarkers.push_back(m);
    }

  //移除那些角点互相离的太近的四边形
  vector< pair<int,int> > tooNearCandidates;
  for(size_t i=0;i<possibleMarkers.size();i++)
    {
      const Marker& m1 = possibleMarkers[i];
      //计算两个maker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近,把这一对四边形放入移除队列。
      for(size_t j=i+1;j<possibleMarkers.size();j++)
        {
          const Marker& m2 = possibleMarkers[j];
          float distSquared = 0;
          for(int c=0;c<4;c++)
            {
              Point v = m1.points[c] - m2.points[c];
              //向量的点乘－》两点的距离
              distSquared += v.dot(v);
            }
          distSquared /= 4;

          if(distSquared < 100)
            {
              tooNearCandidates.push_back(pair<int,int>(i,j));
            }
        }
    }

  //移除了相邻的元素对的标识
  //计算距离相近的两个marker内部，四个点的距离和，将距离和较小的，在removlaMask内做标记，即不作为最终的detectedMarkers 
  vector<bool> removalMask(possibleMarkers.size(),false);

  for(size_t i=0;i<tooNearCandidates.size();i++)
    {
      //求这一对相邻四边形的周长
      float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
      float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

      //谁周长小，移除谁
      size_t removalIndex;
      if(p1 > p2)
        removalIndex = tooNearCandidates[i].second;
      else
        removalIndex = tooNearCandidates[i].first;

      removalMask[removalIndex] = true;
    }

  //返回候选，移除相邻四边形中周长较小的那个，放入待检测的四边形的队列中
  detectedMarkers.clear();
  for(size_t i = 0;i<possibleMarkers.size();i++)
    {
      if(!removalMask[i])
        detectedMarkers.push_back(possibleMarkers[i]);
    }

}

bool MarkerDetector::recognizeMarkers(const Mat& grayscale,vector<Marker>& detectedMarkers)
{
  Mat canonicalMarkerImage;
  //char name[20] = "";

  vector<Marker> goodMarkers;

  for(size_t i=0;i<detectedMarkers.size();i++)
    {
      Marker& marker = detectedMarkers[i];
      //输入原始图像和变换之后的图像的对应4个点，便可以得到变换矩阵
      Mat markerTransform = getPerspectiveTransform(marker.points,m_markerCorners2d);
      //对图像进行透视变换,这就得到和标识图像一致正面的图像
      warpPerspective(grayscale,canonicalMarkerImage,markerTransform,markerSize2d);

      // sprintf(name,"warp_%d.jpg",i);
      // imwrite(name,canonicalMarkerImage);
      //#ifdef SHOW_DEBUG_IMAGES
#if 0
         {
          Mat markerImage = grayscale.clone();
          marker.drawContour(markerImage);
          Mat markerSubImage = markerImage(boundingRect(marker.points));

          imshow("Source marker" + ToString(i),markerSubImage);
          imshow("Marker " + ToString(i),canonicalMarkerImage);
        }
#endif

      int nRotations;
      int id = Marker::getMarkerId(canonicalMarkerImage,nRotations);
      //cout << "ID: " << id << endl;
      
      if(id!=-1)
        {
          marker.id = id;
          //sort the points so that they are always in the same order no matter the camera orientation  
          //Rotates the order of the elements in the range [first,last), in such a way that the element pointed by middle becomes the new first element.
          std::rotate(marker.points.begin(),marker.points.begin() + 4 - nRotations,marker.points.end());//就是一个循环移位

          goodMarkers.push_back(marker);
        }
    }

  //refine using subpixel accuracy the corners  是把所有标识的四个顶点都放在一个大的向量中。
  if(goodMarkers.size() > 0) {
      //找到所有标记的角点 
      vector<Point2f> preciseCorners(4*goodMarkers.size());//每个marker四个点
      for(size_t i=0;i<goodMarkers.size();i++)
        {
          Marker& marker = goodMarkers[i];

          for(int c=0;c<4;c++)
             {
              preciseCorners[i*4+c] = marker.points[c];//i表示第几个marker，c表示某个marker的第几个点
            }
        }

      // 发现亚像素精度的角点位置
      TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,30,0.01);
      cornerSubPix(grayscale,preciseCorners,cvSize(5,5),cvSize(-1,-1),termCriteria);

      //把精准化的坐标传给每一个marker
      for(size_t i=0;i<goodMarkers.size();i++)
         {
          Marker& marker = goodMarkers[i];
            for(int c=0;c<4;c++)
               {
                marker.points[c] = preciseCorners[i*4+c];
                //cout<<"X:"<<marker.points[c].x<<"Y:"<<marker.points[c].y<<endl;
              }
        }

      //画出细化后的矩形图片
      Mat markerCornersMat(grayscale.size(),grayscale.type());
      markerCornersMat = Scalar(0);
      for(size_t i=0;i<goodMarkers.size();i++) {
	goodMarkers[i].drawContour(markerCornersMat,Scalar(255));
      }

      imshow("markers_refined",grayscale*0.5 + markerCornersMat);
      //imshow("gray",grayscale);
      //imshow("gray/2",grayscale*0.5);
      //imshow("markerCorners",markerCornersMat);
      //imwrite("Markers refined edges" + ".png",grayscale*0.5 + markerCornersMat);   
      //imwrite("refine.jpg",grayscale*0.5 + markerCornersMat);

      detectedMarkers = goodMarkers;
  } else {
    return false;
  }
  
  return true;
}

//对每一个标记，求出其相对于相机的转换矩阵。找到上面监测到的标记的位置
void MarkerDetector::estimatePosition(vector<Marker>& detectedMarkers,Mat_<float>& camMatrix,Mat_<float>& distCoeff)
{
  for(size_t i=0;i<detectedMarkers.size();i++) {
    Marker& m = detectedMarkers[i];

    Mat raux,taux;
    solvePnP(m_markerCorners3d,m.points,camMatrix,distCoeff,raux,taux);

    m.R = raux.clone();
    m.T = taux.clone();
  }
}

bool MarkerDetector::findMarkers(const Mat& frame,vector<Marker>& detectedMarkers)
{
  //Mat bgraMat(frame.height,frame.width,CV_8UC4,frame.data,frame.stride);
  prepareImage(frame,m_grayscaleImage);
  performThreshold(m_grayscaleImage,m_thresholdImg);
  findContour(m_thresholdImg,m_contours,m_grayscaleImage.cols/5);
  findCandidates(m_contours,detectedMarkers);
  return recognizeMarkers(m_grayscaleImage,detectedMarkers);
}

cv::Mat Matrix2Euler(cv::Mat matrix)
{
  double PI = 3.141592653;
  //cv::Mat q = Matrix2Quaternion(matrix);
  //cv::Mat angle = Quaternion2Euler(q);
  //return angle.clone();

  float m[4][4] = {0};
  for(int a=0;a<4;a++)
    for(int b=0;b<4;b++)
      m[a][b]=matrix.at<float>(a,b);

  float a[3];
  a[0] = atan2f(m[2][1],m[2][2]) *180/PI;
  a[1] = atan2f(-m[2][0], sqrtf(m[2][1]*m[2][1] +  m[2][2]*m[2][2])) *180/PI;
  a[2] = atan2f(m[1][0], m[0][0]) *180/PI;
  return cv::Mat(3,1,CV_32FC1,a).clone();
}

void showDepth(Mat frame,vector<Point2f> p,Mat R,Mat T)
{
  Mat color(frame.size(),CV_8UC3);
  cvtColor(frame,color,CV_GRAY2BGR);

  if(p.size() != 4)
    return;

  line(color, p[0], p[1], Scalar(0,0,255), 3);
  line(color, p[1], p[2], Scalar(0,0,255), 3);
  line(color, p[2], p[3], Scalar(0,0,255), 3);
  line(color, p[3], p[0], Scalar(0,0,255), 3);

  //画图像中心和marker重心
  line(color, Point2f(frame.cols/2,0), Point2f(frame.cols/2,frame.rows), Scalar(200,0,0), 1);
  line(color, Point2f(0,frame.rows/2), Point2f(frame.cols,frame.rows/2), Scalar(200,0,0), 1);
  Point2f pcenter = (p[0]+p[1]+p[2]+p[3])/4.f;
  circle(color, pcenter, 4, Scalar(0,255,255));

  //Mat t = -R.t()*T;
  Mat a = Matrix2Euler(R);
  Mat t = T;
  
  char str[64];

  //sprintf(str, "Angle: %4.2f %4.2f %4.3f",a.at<float>(0),a.at<float>(1),a.at<float>(2));
  //putText(color, str, Point2f(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
  sprintf(str, "Translation: %4.2f %4.2f %4.2f",t.at<float>(0),t.at<float>(1),t.at<float>(2));
  putText(color, str, Point2f(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    
  imshow("world_Frame_0", color);
  waitKey(10);
}

bool MarkerDetector::processFrame(const Mat& frame,Mat_<float>& camMatrix,Mat_<float>& distCoeff,vector<Marker>& markers)
{
  if(findMarkers(frame,markers)) {
    estimatePosition(markers,camMatrix,distCoeff);
    sort(markers.begin(),markers.end());

    Marker m = markers[0];
    cout<< "R:"<< m.R.t();
    cout<< "T:"<< m.T.t() <<endl;

    showDepth(frame, m.points, m.R, m.T);

    return true;
  }

  return false;
}
