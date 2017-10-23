#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class RGBDCalibrator
{
  int shift_coef;
  int calib_count;
  int stepsize;
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  message_filters::Subscriber<sensor_msgs::Image> color_image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_;
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo,
                                    sensor_msgs::Image> sync_;
  image_transport::Publisher color_image_pub_;
  //image_transport::Publisher color_info_pub_;

public:
  RGBDCalibrator(double sc)
    : shift_coef(sc),
      calib_count(0),
      stepsize(1),
      it_(nh_),
      color_image_sub_(nh_, "camera/color/image_raw", 1),
      color_info_sub_(nh_, "camera/color/camera_info", 1),
      depth_image_sub_(nh_, "camera/depth/image_raw", 1),
      sync_(color_image_sub_, color_info_sub_, depth_image_sub_, 10)
  {
    sync_.registerCallback(boost::bind(&RGBDCalibrator::matchColorDepth, this, _1, _2, _3));
    color_image_pub_ = it_.advertise("camera/rgb/image_raw", 1);//"matched_camera/color/image_raw", 1);
    //color_info_pub_ = it_.advertise("matched_camera/color/camera_info", 1);
  }

  void matchColorDepth(const sensor_msgs::ImageConstPtr& color_image_data,
                       const sensor_msgs::CameraInfoConstPtr& color_info_data,
                       const sensor_msgs::ImageConstPtr& depth_image_data)
  {
  //Shifts color pixels over to match depth image - to correct for distance between RGB camera
  //and depth camera - compares edge detection, and tries to find a local maxima shift by finding
  //the dot product between them
    cv_bridge::CvImagePtr cv_color_image_ptr;
    cv_bridge::CvImagePtr cv_depth_image_ptr;
    try
    {
      color_image_pub_.publish(*color_image_data);
      return;
      cv_color_image_ptr = cv_bridge::toCvCopy(color_image_data,
                                               sensor_msgs::image_encodings::BGR8);
      cv_depth_image_ptr = cv_bridge::toCvCopy(depth_image_data,
                                               sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    //First they must be the same size
    cv::resize(cv_color_image_ptr->image, cv_color_image_ptr->image,
               cv::Size(depth_image_data->width, depth_image_data->height));
    const cv::Mat d_original = cv_depth_image_ptr->image;
    

    //Runs calib_count times - to adjust shift_coef
    if(calib_count)
    {
      //First uses OpenCV to detect edges
      cv::Mat color_edge_detect;
      cv::cvtColor(cv_color_image_ptr->image, color_edge_detect, CV_BGR2GRAY);
      cv::blur(color_edge_detect, color_edge_detect, cv::Size(3, 3));
      cv::Canny(color_edge_detect, color_edge_detect, 50, 150, 3);

      cv::Mat depth_edge_detect;
      cv_depth_image_ptr->image.convertTo(cv_depth_image_ptr->image, CV_8U);
      cv::bilateralFilter(cv_depth_image_ptr->image, depth_edge_detect, 7, 90, 0);
      cv::Canny(depth_edge_detect, depth_edge_detect, 50, 250, 3);
      cv::GaussianBlur(depth_edge_detect, depth_edge_detect, cv::Size(49, 49), 0, 0);
      cv_color_image_ptr->encoding = "mono8";
      cv_color_image_ptr->image = depth_edge_detect;
      color_image_pub_.publish(cv_color_image_ptr->toImageMsg());


      int n_rows = d_original.rows;
      int n_cols = d_original.cols;
      if(d_original.isContinuous())
      {
        n_cols *= n_rows;
        n_rows = 1;
      }

      int i, j;
      const ushort* p;
      int r = 0; int c = 0;
      int max_c = d_original.cols - 1;
      float score = 0;
      float r_score = 0;
      int r_shift_coef = shift_coef + stepsize;
      int new_c, r_new_c;

      //Basically a giant dot product
      for(i = 0; i < n_rows; ++i)
      {
        p = d_original.ptr<ushort>(i);
        for(j = 0; j < n_cols; ++j)
        {
          if(p[j])
          {
            new_c = c + (int) shift_coef/p[j];
            r_new_c = c + (int) r_shift_coef/p[j];
            if(!(r_new_c > max_c || new_c < 0))
            {
              score += (color_edge_detect.at<uchar>(cv::Point(new_c, r))/255.0*
                        depth_edge_detect.at<uchar>(cv::Point(new_c, r))/255.0);
              r_score += (color_edge_detect.at<uchar>(cv::Point(r_new_c, r))/255.0*
                          depth_edge_detect.at<uchar>(cv::Point(r_new_c, r))/255.0);

            }
          }

          c++;
          if(c > max_c)
          {
            c = 0;
            r++;
          }
        }
      }
      if(score > r_score)
      {
        shift_coef -= stepsize;
      }
      else if(r_score > score)
      {
        shift_coef += stepsize;
      }
      ROS_INFO("%d %d %f %f", calib_count, shift_coef, score, r_score);

      calib_count--;

    }
    else
    {
      //Loops through data and makes RGB pixel shifts based on depth - farther it is,
      //the less it shifts
      cv::Mat rgb_original = cv_color_image_ptr->image;
      cv::Mat rgb_calibrated = cv::Mat::zeros(d_original.rows, d_original.cols, CV_8UC3);
      int n_rows = d_original.rows;
      int n_cols = d_original.cols;
      if(d_original.isContinuous())
      {
        n_cols *= n_rows;
        n_rows = 1;
      }

      int i, j;
      const ushort* p;
      int r = 0; int c = 0;
      int max_c = d_original.cols - 1;

      for(i = 0; i < n_rows; ++i)
      {
        p = d_original.ptr<ushort>(i);
      	for(j = 0; j < n_cols; ++j)
      	{
          int new_c = c;
          if(p[j])
          {
            new_c += (int) shift_coef/p[j];
            if(new_c > max_c || new_c < 0)
            {
              rgb_calibrated.at<cv::Vec3b>(cv::Point(c, r)) =
                  cv::Vec3b(0, 0, 0);
            }
            else
            {
              rgb_calibrated.at<cv::Vec3b>(cv::Point(c, r)) =
                  rgb_original.at<cv::Vec3b>(cv::Point(new_c, r));
            }
          }

          c++;
          if(c > max_c)
          {
            c = 0;
            r++;
          }
      	}
      }
      
      cv_color_image_ptr->image = rgb_calibrated;
      color_image_pub_.publish(cv_color_image_ptr->toImageMsg());
    }
  }

  void calibrate(int max_cc, int ss)
  {
    calib_count = max_cc;
    stepsize = ss;
  }
  bool is_calibrating()
  {
    return (bool) calib_count;
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rgbd_calibrator");
  RGBDCalibrator rgbd_cal(-24000);
  if(argc > 1)
    rgbd_cal.calibrate(atoll(argv[1]), 10);
  ros::spin();
  return 0;
}

