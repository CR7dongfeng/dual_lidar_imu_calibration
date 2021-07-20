#include <iostream>
#include <pcl/io/pcd_io.h>
#include "lidar_type.h"
#include "ppk_file_reader.h"
#include "graph_slam_options.h"
#include "calibration_process.h"
#include "scan_input_calib.h"

using namespace jf;

typedef struct PointXYZIT {
	PCL_ADD_POINT4D
	float intensity;
	double timestamp;
	uint16_t ring;                   // laser ring number
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16 PPoint;

// pcl point cloud
POINT_CLOUD_REGISTER_POINT_STRUCT(
		PointXYZIT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
		(double, timestamp, timestamp)(std::uint16_t, ring, ring))
typedef pcl::PointCloud<PPoint> PPointCloud;

/*
 * 搞清楚输入有哪些即可
 * 1、后解算文件，也即IE文件（以前的格式，21列或22列的形式）
 * 2、pcd文件夹，存放由pcap转来的pcd，一般来说，每个pcd点云时间长度为两秒。注意有两个
 * 3、pcd文件夹里面有多少pcd。例如60个pcd，其名称为10.pcd - 69.pcd
 * 4、外参初值
 */
int main(int argc, char** argv) {
	// 1、解析后解算原始数据
	// 输入一：后解算文件
	PpkFileReader ppk("/home/juefx/tasks/2021_calib_new/ie.txt", 1000);
	// 输入二：pcd文件夹
	std::string fileheader_top = "/home/juefx/tasks/2021_calib_new/pcd0/";
  std::string fileheader_back = "/home/juefx/tasks/2021_calib_new/pcd1/";
	// 输入三：定义用于标定的点云组数
	const size_t cloud_num = 100;

	std::vector<PPointCloud::Ptr> cloud_vec_top;
    cloud_vec_top.reserve(cloud_num);
    cloud_vec_top.resize(cloud_num);
    std::vector<PPointCloud::Ptr> cloud_vec_back;
    cloud_vec_back.reserve(cloud_num);
    cloud_vec_back.resize(cloud_num);
	
	std::vector<LidarScanWithOrigin> lidar_scans_with_origin_top;
    lidar_scans_with_origin_top.reserve(cloud_num);
    std::vector<LidarScanWithOrigin> lidar_scans_with_origin_back;
    lidar_scans_with_origin_back.reserve(cloud_num);

	// 2、组织点云数据
	for(size_t i=0; i<cloud_num; i++) {
		cloud_vec_top[i].reset(new PPointCloud());
        cloud_vec_back[i].reset(new PPointCloud());

		std::string filename = std::to_string(i+10) + ".pcd";
		
		std::string file_top = fileheader_top + filename;
        std::string file_back = fileheader_back + filename;
		std::cout << file_top << std::endl;
        std::cout << file_back << std::endl;

		pcl::io::loadPCDFile(file_top, *cloud_vec_top[i]);
        pcl::io::loadPCDFile(file_back, *cloud_vec_back[i]);

		LidarScan lidar_scan_top;
        LidarScan lidar_scan_back;
        lidar_scan_top.reserve(cloud_vec_top[i]->size());
        lidar_scan_back.reserve(cloud_vec_back[i]->size());

		for(size_t j=0; j<cloud_vec_top[i]->size(); j++) {
			LidarPoint lidar_point;
			lidar_point.x = cloud_vec_top[i]->points[j].x;
			lidar_point.y = cloud_vec_top[i]->points[j].y;
			lidar_point.z = cloud_vec_top[i]->points[j].z;
			double sqr_dis = lidar_point.x*lidar_point.x + lidar_point.y*lidar_point.y + lidar_point.z*lidar_point.z;
			if(sqr_dis < 5*5) continue;
			lidar_point.intensity = cloud_vec_top[i]->points[j].intensity;
			lidar_point.secs = static_cast<uint32_t>(cloud_vec_top[i]->points[j].timestamp);
			lidar_point.nsecs = static_cast<uint32_t>((cloud_vec_top[i]->points[j].timestamp - lidar_point.secs) * 1e9);
            lidar_scan_top.emplace_back(std::move(lidar_point));
		}
        for(size_t j=0; j<cloud_vec_back[i]->size(); j++) {
            LidarPoint lidar_point;
            lidar_point.x = cloud_vec_back[i]->points[j].x;
            lidar_point.y = cloud_vec_back[i]->points[j].y;
            lidar_point.z = cloud_vec_back[i]->points[j].z;
          double sqr_dis = lidar_point.x*lidar_point.x + lidar_point.y*lidar_point.y + lidar_point.z*lidar_point.z;
          if(sqr_dis < 5*5) continue;
            lidar_point.intensity = cloud_vec_back[i]->points[j].intensity;
            lidar_point.secs = static_cast<uint32_t>(cloud_vec_back[i]->points[j].timestamp);
            lidar_point.nsecs = static_cast<uint32_t>((cloud_vec_back[i]->points[j].timestamp - lidar_point.secs) * 1e9);
            lidar_scan_back.emplace_back(std::move(lidar_point));
        }

		cout.precision(16);
		double origin_timestamp = (long long unsigned int)(cloud_vec_top[i]->points[0].timestamp * 1000) / 1000.0;
//		cout << "origin_timestamp" << origin_timestamp << endl;
		LidarScanWithOrigin lidar_scan_with_origin_top(origin_timestamp, lidar_scan_top);
        LidarScanWithOrigin lidar_scan_with_origin_back(origin_timestamp, lidar_scan_back);
		lidar_scans_with_origin_top.emplace_back(std::move(lidar_scan_with_origin_top));
        lidar_scans_with_origin_back.emplace_back(std::move(lidar_scan_with_origin_back));
		
		std::cout << "i = " << i << std::endl;
		std::cout << "lidar_scans_with_origin_top.size() in cycle : " << lidar_scans_with_origin_top.size() << std::endl;
        std::cout << "lidar_scans_with_origin_back.size() in cycle : " << lidar_scans_with_origin_back.size() << std::endl;
	}
    cloud_vec_top.clear();
    cloud_vec_back.clear();
	
	// 3、标定选项
	GraphSlamOptions options;
	options.cpu_threads = 10;
	options.lidar_max_radius = 40;
	
	// 4、输入外参初值
	Eigen::Isometry3d lidar_imu_extrinsic = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d lidar_imu_extrinsic_back = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d lidar_lidar_extrinsic = Eigen::Isometry3d::Identity();

    lidar_imu_extrinsic.linear() = (Eigen::AngleAxisd(180*M_PI/180, Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(0*M_PI/180, Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(0*M_PI/180, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    lidar_imu_extrinsic.translation() = Eigen::Vector3d{-0.0045, 0.02534, 0.20771};
    
    lidar_imu_extrinsic_back.linear() = (Eigen::AngleAxisd(0*M_PI/180, Eigen::Vector3d::UnitZ()) *
                                         Eigen::AngleAxisd(0*M_PI/180, Eigen::Vector3d::UnitY()) *
                                         Eigen::AngleAxisd(20*M_PI/180, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    lidar_imu_extrinsic_back.translation() = Eigen::Vector3d{0.1595, -0.482014, 0.040823};

    lidar_lidar_extrinsic = lidar_imu_extrinsic.inverse() * lidar_imu_extrinsic_back;

//  // old
//  lidar_imu_extrinsic.matrix() <<
//                               -0.9998436892216465 , -0.01765894091782343 ,0.0008711653972563576  ,             -0.0045,
//  0.01765972564912217  , -0.9998436484997313, 0.0009014680164672614   ,            0.02534,
//  0.0008551102187974801, 0.0009167116492105747 ,   0.9999992142128242 ,              0.20771,
//  0                 ,     0         ,             0      ,                1;
//  lidar_lidar_extrinsic.matrix() <<
//                                 -0.9999397515934392 , 0.007025873008885489 , 0.005352718824665622 ,  -0.164,
//  -0.004731090011881492 ,  -0.9377889418242196 ,   0.3471143309760622  ,            0.507354,
//  0.007458637787987197 ,   0.3470754659870427 ,   0.9378068397331902  ,           -0.166887,
//  0              ,       0     ,                0      ,               1;

// final
  lidar_imu_extrinsic.matrix() <<
                               -0.9998436668751289 , -0.01766001685364523, 0.0008749938602652187   ,            -0.0045,
    0.01766059589350483  , -0.9998438226583553 ,0.0006585168516606963  ,             0.02534,
    0.0008632277873515036, 0.0006738668166389943,    0.9999994003704705 ,              0.20771,
    0            ,          0        ,              0      ,                1;
  lidar_lidar_extrinsic.linear() << -0.999872  , 0.0093921 ,  0.0111651,
  -0.00501575 ,  -0.939867 ,   0.341443,
  0.013701 ,   0.341351  ,  0.939835;
  lidar_lidar_extrinsic.translation() << -0.169931, 0.521, -0.168807;

#if 0 // 输入旋转矩阵，转换成欧拉角
    {
        Extrinsic xyzrpa = matrix4dToExtrinsic(lidar_imu_extrinsic.matrix());
        Extrinsic xyzrpa_ll = matrix4dToExtrinsic(lidar_lidar_extrinsic.matrix());
        Extrinsic xyzrpa_back = matrix4dToExtrinsic((lidar_imu_extrinsic*lidar_lidar_extrinsic).matrix());

        std::cout << "lidar_to_body Extrinsic : " << std::endl;
        std::cout << "rotation matrix : \n" << lidar_imu_extrinsic.matrix() << std::endl;
        std::cout << "x y z r p a : \n"
                  << xyzrpa.x << " " << xyzrpa.y << " " << xyzrpa.z << " "
                  << xyzrpa.r << " " << xyzrpa.p << " " << xyzrpa.a << std::endl;

        std::cout << "back_lidar_to_body Extrinsic : " << std::endl;
        std::cout << "rotation matrix : \n" << (lidar_imu_extrinsic*lidar_lidar_extrinsic).matrix() << std::endl;
        std::cout << "x y z r p a : \n"
                  << xyzrpa_back.x << " " << xyzrpa_back.y << " " << xyzrpa_back.z << " "
                  << xyzrpa_back.r << " " << xyzrpa_back.p << " " << xyzrpa_back.a << std::endl;

        return 0;
    }
#endif

    // 5、进入标定程序
	auto pair = calibrationProcess(
			lidar_scans_with_origin_top,
            lidar_scans_with_origin_back,
			ppk,
            lidar_imu_extrinsic,
            lidar_lidar_extrinsic,
            options);

    lidar_imu_extrinsic = pair.first;
    lidar_lidar_extrinsic = pair.second;

    Extrinsic xyzrpa = matrix4dToExtrinsic(lidar_imu_extrinsic.matrix());
    Extrinsic xyzrpa_ll = matrix4dToExtrinsic(lidar_lidar_extrinsic.matrix());
    Extrinsic xyzrpa_back = matrix4dToExtrinsic((lidar_imu_extrinsic*lidar_lidar_extrinsic).matrix());

    std::cout << std::endl << "------------------FINAL OUTPUT------------" << std::endl << std::endl;
    std::cout << "lidar_to_body Extrinsic : " << std::endl;
    std::cout << "rotation matrix : \n" << lidar_imu_extrinsic.matrix() << std::endl;
    std::cout << "x y z r p a : \n"
                        << xyzrpa.x << " " << xyzrpa.y << " " << xyzrpa.z << " "
                        << xyzrpa.r << " " << xyzrpa.p << " " << xyzrpa.a << std::endl;

    std::cout << "lidar_to_lidar Extrinsic : " << std::endl;
    std::cout << "rotation matrix : \n" << lidar_lidar_extrinsic.matrix() << std::endl;
    std::cout << "x y z r p a : \n"
              << xyzrpa_ll.x << " " << xyzrpa_ll.y << " " << xyzrpa_ll.z << " "
              << xyzrpa_ll.r << " " << xyzrpa_ll.p << " " << xyzrpa_ll.a << std::endl;

    std::cout << "back_lidar_to_body Extrinsic : " << std::endl;
    std::cout << "rotation matrix : \n" << (lidar_imu_extrinsic*lidar_lidar_extrinsic).matrix() << std::endl;
    std::cout << "x y z r p a : \n"
              << xyzrpa_back.x << " " << xyzrpa_back.y << " " << xyzrpa_back.z << " "
              << xyzrpa_back.r << " " << xyzrpa_back.p << " " << xyzrpa_back.a << std::endl;

    ofstream ofs("extrinsic.txt");
    ofs << xyzrpa.x << " " << xyzrpa.y << " " << xyzrpa.z << " "
        << xyzrpa.r << " " << xyzrpa.p << " " << xyzrpa.a << endl;
    ofs << xyzrpa_ll.x << " " << xyzrpa_ll.y << " " << xyzrpa_ll.z << " "
        << xyzrpa_ll.r << " " << xyzrpa_ll.p << " " << xyzrpa_ll.a << endl;
    ofs << xyzrpa_back.x << " " << xyzrpa_back.y << " " << xyzrpa_back.z << " "
        << xyzrpa_back.r << " " << xyzrpa_back.p << " " << xyzrpa_back.a << endl;


	return 0;
}

