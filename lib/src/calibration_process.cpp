//
// Created by ubuntu on 19-7-3.
//
#include "calibration_process.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <future>
#include <vector>
#include "calib_options.h"
#include "calibration_once.h"
#include "scan_input_calib.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>


#include "graph_slam_options.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>

#include <fstream>

namespace jf {

std::pair<Eigen::Isometry3d, Eigen::Isometry3d> calibrationProcess(
		const vector<DeformedLidarScan>& deformed_scans_top,
        const vector<DeformedLidarScan>& deformed_scans_back,
        const PpkFileReader& ppk,
		const Eigen::Isometry3d& extrinsic_input,
		const Eigen::Isometry3d& extrinsic_ll_input,
		const GraphSlamOptions& options) {
	Eigen::Isometry3d out_extrinsic = extrinsic_input;
    Eigen::Isometry3d out_extrinsic_ll = extrinsic_ll_input;

	std::cout << "initial extrinsic lidar-body:\n" << out_extrinsic.matrix() << endl;
    std::cout << "initial extrinsic lidar-lidar:\n" << out_extrinsic_ll.matrix() << endl;

	CalibrationOptions calib_options;
	
//	calib_options.calib_stage = 2;

	int iterations = 0;

	// 主循环
	while (iterations <= calib_options.max_iterations) {
		// 1、以IMU为原点，将点云数据转换到ENU坐标系，并进行运动补偿
		Eigen::Isometry3d ex_top_to_body = out_extrinsic;
        Eigen::Isometry3d ex_back_to_body = out_extrinsic * out_extrinsic_ll;

		auto local_scans_top = transformLidarScansToIMUOrigin(
				deformed_scans_top, ppk, ex_top_to_body.matrix(), options.cpu_threads);
        auto local_scans_back = transformLidarScansToIMUOrigin(
                deformed_scans_back, ppk, ex_back_to_body.matrix(), options.cpu_threads);

		LOG(INFO) << "size of local_scans_top " << local_scans_top.size();
        LOG(INFO) << "size of local_scans_back " << local_scans_back.size();

		// 2、计算IMU坐标系下的相对运动
		vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
				vec_relative_imu;

		for (size_t i = 0; i < local_scans_top.size(); i++) {
			Eigen::Isometry3d relative_imu = getRelativeTransPoseOfIMU(
					local_scans_top.at(0).pose, local_scans_top.at(i).pose);
			vec_relative_imu.emplace_back(relative_imu);
		}
		
		// 3、将运动补偿后的点云从ENU坐标系转回到Lidar坐标系下
		vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> local_lidar_scans_top;
        vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> local_lidar_scans_back;
		
		for (size_t i = 0; i < local_scans_top.size(); i++) {
            local_lidar_scans_top.emplace_back(std::move(
					transformLocalScanToLidarOrigin(local_scans_top[i], &ex_top_to_body.matrix())
					));
		}
        for (size_t i = 0; i < local_scans_back.size(); i++) {
            local_lidar_scans_back.emplace_back(std::move(
                    transformLocalScanToLidarOrigin(local_scans_back[i], &ex_back_to_body.matrix())
            ));
        }

		local_scans_top.clear();
        local_scans_back.clear();

		LOG(INFO) << "outer iterations = " << iterations;
		std::cout << "before calibration lidar_to_body: \n"
		          << out_extrinsic.matrix() << std::endl;
        std::cout << "before calibration lidar_to_lidar: \n"
                  << out_extrinsic_ll.matrix() << std::endl;

		Eigen::Quaterniond q_former(out_extrinsic.linear());
		Eigen::Vector3d t_former(out_extrinsic.translation());
        Eigen::Quaterniond q_ll_former(out_extrinsic_ll.linear());
        Eigen::Vector3d t_ll_former(out_extrinsic_ll.translation());
#if 0 // 若设置为1，则只用来验证效果
        {
            int size = local_lidar_scans_top.size();
            for(int i=0; i<size; ++i) {
              std::string dir = "/home/juefx/tasks/2021_calib_new/test15/";
                pcl::io::savePCDFileBinary( dir + to_string(i) + "_first_top.pcd", *local_lidar_scans_top[i]);

              Eigen::Isometry3d temp_transform = Eigen::Isometry3d::Identity();
              pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);

              temp_transform = out_extrinsic_ll;
              pcl::transformPointCloud(*local_lidar_scans_back[i], *temp_cloud, temp_transform.matrix());
              pcl::io::savePCDFileBinary(dir + to_string(i) + "_first_back.pcd", *temp_cloud);

              if(i == size-1) break;
              Eigen::Isometry3d move = Eigen::Isometry3d::Identity();

              move = vec_relative_imu[i].inverse() * vec_relative_imu[i + 1];
              temp_transform = out_extrinsic.inverse()*move*out_extrinsic;
              pcl::transformPointCloud(*local_lidar_scans_top[i+1], *temp_cloud, temp_transform.matrix());
              pcl::io::savePCDFileBinary(dir + to_string(i) + "_second_top.pcd", *temp_cloud);

              temp_transform = out_extrinsic.inverse()*move*out_extrinsic*out_extrinsic_ll;
              pcl::transformPointCloud(*local_lidar_scans_back[i+1], *temp_cloud, temp_transform.matrix());
              pcl::io::savePCDFileBinary(dir + to_string(i) + "_second_back.pcd", *temp_cloud);

            }

            std::cout << "PRINT END!!" << std::endl;
            return {out_extrinsic, out_extrinsic_ll};


        }
#endif
		// 4、进行标定并判断阈值，决定是否继续
        auto ex_pair = calibrationOnce(local_lidar_scans_top,
                                       local_lidar_scans_back,
                                          vec_relative_imu,
		                                  out_extrinsic,
		                                  out_extrinsic_ll,
		                                  calib_options);
        out_extrinsic = ex_pair.first;
        out_extrinsic_ll = ex_pair.second;

        calib_options.first_time = false;

		Eigen::Quaterniond q_latter(out_extrinsic.linear());
		Eigen::Vector3d t_latter(out_extrinsic.translation());
        Eigen::Quaterniond q_ll_latter(out_extrinsic_ll.linear());
        Eigen::Vector3d t_ll_latter(out_extrinsic_ll.translation());

        double rot_residual_lidar_to_body =
                fabs(1 - ((q_former * q_latter.conjugate()).normalized()).w());
        std::cout << "outer: rot_residual_lidar_to_body is " << rot_residual_lidar_to_body << endl;

        double rot_residual_lidar_to_lidar =
                fabs(1 - ((q_ll_former * q_ll_latter.conjugate()).normalized()).w());
        std::cout << "outer: rot_residual_lidar_to_lidar is " << rot_residual_lidar_to_lidar << endl;

        double rot_residual = (rot_residual_lidar_to_body + rot_residual_lidar_to_lidar) / 2;
        std::cout << "outer: rot_residual is " << rot_residual << endl;

        double trans_residual_lidar_to_body = (t_former - t_latter).norm();
        std::cout << "outer: trans_residual_lidar_to_body is " << trans_residual_lidar_to_body << endl;
        double trans_residual_lidar_to_lidar = (t_former - t_latter).norm();
        std::cout << "outer: trans_residual_lidar_to_lidar is " << trans_residual_lidar_to_lidar << endl;
        double trans_residual = (trans_residual_lidar_to_body + trans_residual_lidar_to_lidar) / 2;
        std::cout << "outer: trans_residual is " << trans_residual << endl;

		if (calib_options.calib_stage == ICP_Ceres::CALIB_R_STEP) {
			if (rot_residual < calib_options.outer_rotation_epsilon) {
				LOG(INFO) << "Extrinsic R converges! Iterations is :" << iterations;
				break;
			} else if (iterations == calib_options.max_iterations) {
				LOG(INFO) << "Extrinsic R - max iterations!";
				break;
			}
		} else if (calib_options.calib_stage == ICP_Ceres::CALIB_XY_STEP) {
//			if (trans_residual < calib_options.outer_translation_epsilon) {
//				LOG(INFO) << "Extrinsic XY converges! Iterations is :" << iterations;
//				std::cout << "final extrinsic: \n"
//				          << out_extrinsic.matrix() << std::endl;
//
//				Extrinsic xyzrpa = matrix4dToExtrinsic(out_extrinsic.matrix());
//				std::cout << "x y z r p a : \n"
//									<< xyzrpa.x << " " << xyzrpa.y << " " << xyzrpa.z << " "
//									<< xyzrpa.r << " " << xyzrpa.p << " " << xyzrpa.a << endl;
//				ofstream ofs("/home/limingbo/data/extrinsic.txt");
//				ofs << xyzrpa.x << " " << xyzrpa.y << " " << xyzrpa.z << " "
//				    << xyzrpa.r << " " << xyzrpa.p << " " << xyzrpa.a << endl;
//				return out_extrinsic;
//			}
		}
		iterations++;

	}

	return std::make_pair(out_extrinsic, out_extrinsic_ll);
}
}

/*

 */