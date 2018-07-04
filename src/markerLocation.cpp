//-----------------------------------------
//AR 基于标识，by zhuzhu 2015.11.18
//-----------------------------------------

#include <stdio.h>
#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "MarkerDetector.hpp"

using namespace std;
using namespace cv;

MarkerDetector *pMarkerDetector = NULL;

Mat_<float>  camMatrix;
Mat_<float>  distCoeff;

ros::Publisher pub_odom;
ros::Publisher pub_marker;
ros::Publisher pub_tf;

//旋转矩阵得到四元数
cv::Mat Matrix2Quaternion(cv::Mat matrix)
{
  float tr, qx, qy, qz, qw;

  // 计算矩阵轨迹
  float a[4][4] = {0};
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      a[i][j]=matrix.at<float>(i,j);
  
  // I removed + 1.0f; see discussion with Ethan
  float trace = a[0][0] + a[1][1] + a[2][2]; 
  if( trace > 0 ) {
    // I changed M_EPSILON to 0
    float s = 0.5f / sqrtf(trace+ 1.0f);
    qw = 0.25f / s;
    qx = ( a[2][1] - a[1][2] ) * s;
    qy = ( a[0][2] - a[2][0] ) * s;
    qz = ( a[1][0] - a[0][1] ) * s;
  } else {
    if ( a[0][0] > a[1][1] && a[0][0] > a[2][2] ) {
      float s = 2.0f * sqrtf( 1.0f + a[0][0] - a[1][1] - a[2][2]);
      qw = (a[2][1] - a[1][2] ) / s;
      qx = 0.25f * s;
      qy = (a[0][1] + a[1][0] ) / s;
      qz = (a[0][2] + a[2][0] ) / s;
    } else if (a[1][1] > a[2][2]) {
      float s = 2.0f * sqrtf( 1.0f + a[1][1] - a[0][0] - a[2][2]);
      qw = (a[0][2] - a[2][0] ) / s;
      qx = (a[0][1] + a[1][0] ) / s;
      qy = 0.25f * s;
      qz = (a[1][2] + a[2][1] ) / s;
    } else {
      float s = 2.0f * sqrtf( 1.0f + a[2][2] - a[0][0] - a[1][1] );
      qw = (a[1][0] - a[0][1] ) / s;
      qx = (a[0][2] + a[2][0] ) / s;
      qy = (a[1][2] + a[2][1] ) / s;
      qz = 0.25f * s;
    }    
  }

  float q[] = {qw,qx,qy,qz};
  //cout<< "\n quaternion:"<<cv::Mat(4,1,CV_32FC1,q).t()<<endl;
  return cv::Mat(4,1,CV_32FC1,q).clone();
}

void pubTF(std_msgs::Header header,Mat Q,Mat t)
{
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    // camera frame
    transform.setOrigin(tf::Vector3(t.at<float>(0),t.at<float>(1),t.at<float>(2)));
    
    q.setW(Q.at<float>(0));
    q.setX(Q.at<float>(1));
    q.setY(Q.at<float>(2));
    q.setZ(Q.at<float>(3));
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "camera"));
}


void publishResult(vector<Point3f> markerCorners, Mat Rvec, Mat T)
{
  Mat RMat,J;
  Rodrigues(Rvec,RMat,J);
  Mat q = Matrix2Quaternion(RMat);
  Mat t = T/1000.f; //mm to m
  
  nav_msgs::Odometry odometry;
  odometry.header.frame_id = "world";
  odometry.header.stamp = ros::Time::now();
  odometry.pose.pose.position.x = t.at<float>(0);
  odometry.pose.pose.position.y = t.at<float>(1);
  odometry.pose.pose.position.z = t.at<float>(2);
  odometry.pose.pose.orientation.x = q.at<float>(1);
  odometry.pose.pose.orientation.y = q.at<float>(2);
  odometry.pose.pose.orientation.z = q.at<float>(3);
  odometry.pose.pose.orientation.w = q.at<float>(0);
  pub_odom.publish(odometry);

  sensor_msgs::PointCloud pd;
  pd.header.frame_id = "world";
  pd.header.stamp = ros::Time::now();
  for(int i=0;i<markerCorners.size();i++) {
    geometry_msgs::Point32 p;
    p.x = markerCorners[i].x / 1000.f;
    p.y = markerCorners[i].y / 1000.f;
    p.z = markerCorners[i].z / 1000.f;
    pd.points.push_back(p);
  }
  pub_marker.publish(pd);

  pubTF(odometry.header,q,t);
}


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv_bridge::CvImageConstPtr ptr;
  
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else {
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  
  
  cv::Mat src = ptr->image;
  vector<Marker> markers;
  if(pMarkerDetector->processFrame(src,camMatrix, distCoeff,markers)) {
    publishResult(pMarkerDetector->m_markerCorners3d, markers[0].R, markers[0].T);
  }
  
}

int main(int argc,char *argv[])
{
  ros::init(argc, argv, "marker_location");
  ros::NodeHandle n("~");

  // 导入相机内参数
  double intrinsics[9] = {542.0642361133936, 0, 321.19682257337666,
			  0, 541.1428792728802, 209.5818251264706,
			  0, 0, 1};
  double dist_coeff[4] = {0.09198244279760188, -0.14495513280487463, -0.0009522199688690519,
			  -0.0026345955272592514};
  camMatrix = Mat(3, 3, CV_64F,intrinsics);
  distCoeff = Mat(4, 1, CV_64F,dist_coeff);
  cout << camMatrix << endl;
  cout << distCoeff << endl;

  // 设置marker大小
  float markerSize3d = 56.f; //157mm
  pMarkerDetector = new MarkerDetector(markerSize3d);

  pub_odom = n.advertise<nav_msgs::Odometry>("/odom", 1);
  pub_marker = n.advertise<sensor_msgs::PointCloud>("/marker", 1000);
    
  ros::Subscriber sub_img = n.subscribe("/cam0/image_raw", 1, img_callback);

  
  ros::spin();
  delete pMarkerDetector;
  return 0;
}

