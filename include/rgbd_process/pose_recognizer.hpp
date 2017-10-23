#include <ros_caffe/caffe.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>
#include <memory>

class PoseRecognizer
{
private:
  std::shared_ptr<caffe::Net<float> > net_sp_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<std::string> labels_;
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ros::Subscriber image_sub_;
//  message_filters::Subscriber<sensor_msgs::Image> color_image_sub_;
//  message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub_;
//  message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_;
//  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo,
//                                    sensor_msgs::Image> sync_;
//  image_transport::Publisher color_image_pub_;

  
public:
  PoseRecognizer(const std::string& model_file, const std::string& trained_file,
                 const std::string& img_topic);
  void recognize(const sensor_msgs::ImageConstPtr& color_image_data);
  void poseExtract(const cv::Mat& img);
  void uCharCvMatToFloatPtr(float* floatImage, const cv::Mat& cvImage, const bool normalize);
  unsigned long shapeSize(const std::vector<int>& shape, const int& unitSize);
};
  
    
