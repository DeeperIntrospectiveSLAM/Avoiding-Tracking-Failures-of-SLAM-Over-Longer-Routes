#include <ros/ros.h>
#include <iostream>
#include <string>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <iomanip>

using namespace std;

int main(int argc, char* argv[])
{
    ros::init(argc,argv,"minos_image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher            pub=it.advertise("/camera/image_raw",12);
    int i = 0;
    cv::Mat image;
    cv::Mat imager;
    sensor_msgs::ImagePtr msg;
    ros::Rate loop_rate(15);

    while(nh.ok())
    {
	stringstream new_name;
	new_name << std::setw(10) << std::setfill('0') << i;
	string f_name = new_name.str();

        stringstream ss;
       // ss << "/home/frames/color_.jpg";
       // std::cout << "I copied\n";         
        ss << "/home/rgbd_dataset_freiburg1_xyz/rgb/" << f_name << ".png";
       // std::cout << "I copied\n"; 
        std::string image_name;
        image_name =  ss.str();
        image = cv::imread(image_name,CV_LOAD_IMAGE_COLOR);
        
        if(!image.empty())
        {
            resize(image, imager, cv::Size(640, 480), 0, 0, CV_INTER_AREA);
            //resize(image, imager, cv::Size(1241, 376), 0, 0, CV_INTER_AREA);
            msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",imager).toImageMsg();
            pub.publish(msg);
            std::cout << "I saved\n";
            i++;
        }
        //ROS_INFO_STREAM(image_name);

    	ros::spinOnce();
    	loop_rate.sleep();
    }
}
