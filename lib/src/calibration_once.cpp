#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/pcd_io.h>

#include "calib_options.h"
#include "calibration_once.h"
#include "frame.h"

namespace jf {

void createFrames(
		std::vector<std::shared_ptr<Frame>> &frames,
		const vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &local_lidar_scans_top,
        const vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &local_lidar_scans_back,
		const vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
		&vec_relative_imu,
		const CalibrationOptions &calib_options) {
	if (local_lidar_scans_top.size() != vec_relative_imu.size() ||
	    local_lidar_scans_back.size() != vec_relative_imu.size()) {
		LOG(FATAL) << "local_lidar_scans.size() != vec_relative_imu.size()";
	}
	size_t size = local_lidar_scans_back.size();
	
	vector<pcl::PointCloud<pcl::PointNormal>::Ptr> processed_local_lidar_scans_top;
	for (size_t i = 0; i < local_lidar_scans_top.size(); i++) {
		processed_local_lidar_scans_top.emplace_back(
				cloudPreProcess(local_lidar_scans_top[i]));
	}

    vector<pcl::PointCloud<pcl::PointNormal>::Ptr> processed_local_lidar_scans_back;
    for (size_t i = 0; i < local_lidar_scans_back.size(); i++) {
        processed_local_lidar_scans_back.emplace_back(
                cloudPreProcess(local_lidar_scans_back[i]));
    }
  {
    int sz = processed_local_lidar_scans_back.size();
    for (int i = 0; i < size; ++i) {
      pcl::io::savePCDFileBinary("/home/juefx/tasks/2021_calib_new/test_filter/" + to_string(i) + "_pcl0.pcd",
                                 *processed_local_lidar_scans_top[i]);
      pcl::io::savePCDFileBinary("/home/juefx/tasks/2021_calib_new/test_filter/" + to_string(i) + "_pcl1.pcd",
                                 *processed_local_lidar_scans_back[i]);
    }
  }

  std::cout << "PRINT END!!" << std::endl;
	
	for (size_t i = 0; i < size - 1; i++) {
		Eigen::Isometry3d relative_imu = Eigen::Isometry3d::Identity();
		relative_imu = vec_relative_imu[i].inverse() * vec_relative_imu[i + 1];
		if (relative_imu.translation().norm() > calib_options.min_move_distance) {
			std::shared_ptr<Frame> f(new Frame());
			f->move_t_ = relative_imu;
			f->cloud1_top_ = processed_local_lidar_scans_top[i];
            f->cloud1_back_ = processed_local_lidar_scans_back[i];
			f->cloud2_top_ = processed_local_lidar_scans_top[i+1];
			f->cloud2_back_ = processed_local_lidar_scans_back[i+1];
			f->copyToEigen();
			
			f->cloud1_top_.reset();
			f->cloud1_back_.reset();
            f->cloud2_top_.reset();
            f->cloud2_back_.reset();
			frames.push_back(f);
		}
	}
	
	cout << "frames.size()" << frames.size() << endl;
}

std::pair<Eigen::Isometry3d, Eigen::Isometry3d> calibrationOnce(
		vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &local_lidar_scans_top,
        vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &local_lidar_scans_back,
		const vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
		&vec_relative_imu,
		const Eigen::Isometry3d &extrinsic_input,
        const Eigen::Isometry3d &extrinsic_ll_input,
		const CalibrationOptions &calib_options) {
    Frame::extrinsic_ = extrinsic_input;
    Frame::back_to_top_extrinsic_ = extrinsic_ll_input;
	// 对frames进行初始化
	std::vector<std::shared_ptr<Frame>> frames;
	createFrames(frames, local_lidar_scans_top, local_lidar_scans_back, vec_relative_imu, calib_options);
    local_lidar_scans_top.clear();
    local_lidar_scans_back.clear();

	double thresh = calib_options.first_time ? calib_options.max_correspondence_distance_upper : calib_options.max_correspondence_distance_lower;
	int iteration_counter = 0;
	
//	for (auto f : frames) {
//		f->filterPtsByCorrespondences(1.5 * thresh);
//	}
	// 开始标定
	for (int i = 0; i < calib_options.max_iterations; i++) {
		// 1、对frame进行滤波
		for (auto f : frames) {
		    f->resetKDTree();
		    if(i!=0)  f->refreshUnifiedPts();
			f->getClosestPoints(thresh);

			f->filterCorrespondences();
			f->filterCorrespondencesByNormal();

      vector<Eigen::Vector3d> unified_pts1_local(f->unified_pts1_.begin(), f->unified_pts1_.begin() + f->num1_top_);
      swap(unified_pts1_local, f->unified_pts1_);
      f->resetKDTree();

      f->getClosestPoints(unified_pts1_local, thresh, f->num1_top_, unified_pts1_local.size()-1);

#if 0 // 若设置为1，则用来打印匹配的不同类型
      int num = f->corr_vec_.size();
            int num_top_top = 0, num_top_back = 0, num_back_top = 0, num_back_back = 0;
            int num_ground_top_top = 0, num_ground_top_back = 0, num_ground_back_top = 0, num_ground_back_back = 0;
            int num_local = 0;
            int num_local_ground = 0;

            int num_dis[5] = {0};

            for(const auto& corr : f->corr_vec_) {
                int _dis = corr.dis / 10;
                ++num_dis[_dis];

                if(corr.local_flag) {
                  ++num_local;
                  if(corr.is_ground_first && corr.is_ground_second) ++num_local_ground;
                }
                if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == TOP_LIDAR ) {
                  ++num_top_top;
                  if(corr.is_ground_first && corr.is_ground_second) ++num_ground_top_top;
                }
                if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == BACK_LIDAR ) {
                  ++num_top_back;
                  if(corr.is_ground_first && corr.is_ground_second) ++num_ground_top_back;
                }
                if(corr.lidar_id_first == BACK_LIDAR && corr.lidar_id_second == TOP_LIDAR ) {
                  ++num_back_top;
                  if(corr.is_ground_first && corr.is_ground_second) ++num_ground_back_top;
                }
                if(corr.lidar_id_first == BACK_LIDAR && corr.lidar_id_second == BACK_LIDAR ) {
                  ++num_back_back;
                  if(corr.is_ground_first && corr.is_ground_second) ++num_ground_back_back;
                }
            }
            int now_precision = cout.precision();
            cout.precision(4);
            cout << "corr num is : " << f->corr_vec_.size() << ", ("
                 << double(num_top_top)/num*100 << "%, "
                 << double(num_top_back)/num*100 << "%, "
                 << double(num_back_top)/num*100 << "%, "
                 << double(num_back_back)/num*100 << "%)" << endl;

            cout << "ground : " << "("
              << double(num_ground_top_top)/num*100 << "%, "
              << double(num_ground_top_back)/num*100 << "%, "
              << double(num_ground_back_top)/num*100 << "%, "
              << double(num_ground_back_back)/num*100 << "%)" << endl;

            cout << "local : " << "("
              << double(num_local)/num*100 << "%, "
              << double(num_local_ground)/num*100 << "%)" << endl;

            cout << "dis : " << "("
              << double(num_dis[0])/num*100 << "%, "
              << double(num_dis[1])/num*100 << "%, "
              << double(num_dis[2])/num*100 << "%, "
              << double(num_dis[3])/num*100 << "%, "
              << double(num_dis[4])/num*100 << "%)" << endl;

            cout.precision(now_precision);
#endif




		}
		
		// 2、进行ceres优化
		Eigen::Quaterniond q_former(Frame::extrinsic_.linear());
		Eigen::Vector3d t_former(Frame::extrinsic_.translation());
        Eigen::Quaterniond q_ll_former(Frame::back_to_top_extrinsic_.linear());
        Eigen::Vector3d t_ll_former(Frame::back_to_top_extrinsic_.translation());
		
		ICP_Ceres::ceresOptimizer(frames, true, true, calib_options.calib_stage);
		
		Eigen::Quaterniond q_latter(Frame::extrinsic_.linear());
		Eigen::Vector3d t_latter(Frame::extrinsic_.translation());
        Eigen::Quaterniond q_ll_latter(Frame::back_to_top_extrinsic_.linear());
        Eigen::Vector3d t_ll_latter(Frame::back_to_top_extrinsic_.translation());
		
		std::cout << "iterator: i = " << i << std::endl;
		std::cout << "thresh = " << thresh << std::endl;
		
		// 3、判断阈值并决定是否继续
		double rot_residual_lidar_to_body =
				fabs(1 - ((q_former * q_latter.conjugate()).normalized()).w());
		std::cout << "inner: rot_residual_lidar_to_body is " << rot_residual_lidar_to_body << endl;

        double rot_residual_lidar_to_lidar =
                fabs(1 - ((q_ll_former * q_ll_latter.conjugate()).normalized()).w());
        std::cout << "inner: rot_residual_lidar_to_lidar is " << rot_residual_lidar_to_lidar << endl;

        double rot_residual = (rot_residual_lidar_to_body + rot_residual_lidar_to_lidar) / 2;
        std::cout << "inner: rot_residual is " << rot_residual << endl;

        double trans_residual_lidar_to_body = (t_former - t_latter).norm();
        std::cout << "inner: trans_residual_lidar_to_body is " << trans_residual_lidar_to_body << endl;
        double trans_residual_lidar_to_lidar = (t_ll_former - t_ll_latter).norm();
        std::cout << "inner: trans_residual_lidar_to_lidar is " << trans_residual_lidar_to_lidar << endl;
		double trans_residual = (trans_residual_lidar_to_body + trans_residual_lidar_to_lidar) / 2;
		std::cout << "inner: trans_residual is " << trans_residual << endl;
		
		iteration_counter++;
		
		bool R_converge = ((calib_options.calib_stage == ICP_Ceres::CALIB_R_STEP) &&
		                   (rot_residual < calib_options.rotation_epsilon));

//		R_converge = false;

		bool t_converge =
				((calib_options.calib_stage == ICP_Ceres::CALIB_XY_STEP) &&
				 (trans_residual < calib_options.translation_epsilon));
		
		if (iteration_counter >= calib_options.delay_iterations || R_converge ||
		    t_converge) {
			thresh -= calib_options.distance_step;
			iteration_counter = 0;
			
			if (thresh < calib_options.max_correspondence_distance_lower -
			             calib_options.distance_step / 2) {
				if (R_converge || t_converge)
					break;
				else
					thresh = calib_options.max_correspondence_distance_lower;
			}
		}
	}
	
	std::cout << "calibration_once: return final extrinsic is :\n"
	          << "lidar_to_body : \n" << Frame::extrinsic_.matrix() << std::endl
	          << "lidar_to_lidar : \n" << Frame::back_to_top_extrinsic_.matrix() << std::endl;
	
	return std::make_pair(Frame::extrinsic_, Frame::back_to_top_extrinsic_);
}
}

