#include "frame.h"
#include <algorithm>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>




namespace jf {
Eigen::Isometry3d Frame::extrinsic_;
Eigen::Isometry3d Frame::back_to_top_extrinsic_;

//bool Frame::loadCloud(const string &filename, int whichCLD) {
//	bool ret = true;
//
//	if (whichCLD == 1) {
//		pcl::io::loadPCDFile(filename, *cloud1_);
//	} else if (whichCLD == 2) {
//		pcl::io::loadPCDFile(filename, *cloud2_);
//	} else {
//		ret = false;
//		std::cout << "Parameter error!" << std::endl;
//	}
//
//	return ret;
//}

//void Frame::loadMoveT(const string &filename) {
//	cv::FileStorage fs(filename, cv::FileStorage::READ);
//	cv::Mat tmpmat = cv::Mat::eye(4, 4, CV_64FC1);
//	fs["deltaIMU matrix T"] >> tmpmat;
//
//	cv::cv2eigen(tmpmat, move_t_.matrix());
//}

//void Frame::calculateNormal() {
//	PointCloud::Ptr cloud1_(new PointCloud);
//	PointCloud::Ptr cloud2_(new PointCloud);
//
//	pcl::NormalEstimation<PointT, PointT> norm_est;
//	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
//	norm_est.setSearchMethod(tree);
//	norm_est.setKSearch(10);
//	norm_est.setInputCloud(cloud1_);
//	norm_est.compute(*cloud1_);
//	norm_est.setInputCloud(cloud2_);
//	norm_est.compute(*cloud2_);
//
//	std::vector<int> indices;
//	pcl::removeNaNFromPointCloud(*cloud1_, *cloud1_, indices);
//	pcl::removeNaNFromPointCloud(*cloud2_, *cloud2_, indices);
//}

//void Frame::filter(bool downsampling, bool outlier_filter, bool removeNaN) {
//	if (downsampling) {
//		pcl::VoxelGrid<PointT> grid;
//
//		grid.setLeafSize(0.1, 0.1, 0.1);
//		grid.setInputCloud(cloud1_);
//		grid.filter(*cloud1_);
//		grid.setInputCloud(cloud2_);
//		grid.filter(*cloud2_);
//	}
//
//	if (outlier_filter) {
//		pcl::StatisticalOutlierRemoval<PointT> sor;
//		sor.setMeanK(50);
//		sor.setStddevMulThresh(1.0);
//		sor.setInputCloud(cloud1_);
//		sor.filter(*cloud1_);
//		sor.setInputCloud(cloud2_);
//		sor.filter(*cloud2_);
//	}
//
//	if (removeNaN) {
//		std::vector<int> indices;
//		pcl::removeNaNFromPointCloud(*cloud1_, *cloud1_, indices);
//		pcl::removeNaNFromPointCloud(*cloud2_, *cloud2_, indices);
//	}
//}

//void Frame::filterCurvature() {
//	pcl::ConditionOr<pcl::PointNormal>::Ptr curva_cond(
//			new pcl::ConditionOr<pcl::PointNormal>);
//	curva_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
//			new pcl::FieldComparison<pcl::PointNormal>(
//					"curvature", pcl::ComparisonOps::LT, 0.07)));
//	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
//	condrem.setCondition(curva_cond);
//	condrem.setInputCloud(this->cloud1_);
//	condrem.filter(*(this->cloud1_));
//	condrem.setInputCloud(this->cloud2_);
//	condrem.filter(*(this->cloud2_));
//
//	pcl::StatisticalOutlierRemoval<PointT> sor;
//	sor.setMeanK(10);
//	sor.setStddevMulThresh(1.0);
//	sor.setInputCloud(cloud1_);
//	sor.filter(*cloud1_);
//	sor.setInputCloud(cloud2_);
//	sor.filter(*cloud2_);
//}

// 调用之前必须先更新好move_t和外参
void Frame::copyToEigen() {
    pts1_.clear();
    pts2_.clear();
    nor1_.clear();
    nor2_.clear();

    lidar_id_vec1_.clear();
    lidar_id_vec2_.clear();

    num1_top_ = cloud1_top_->points.size();
    num1_back_ = cloud1_back_->points.size();
    num2_top_ = cloud2_top_->points.size();
    num2_back_ = cloud2_back_->points.size();

    for(auto& point : cloud1_top_->points) {
        pts1_.push_back(Eigen::Vector3d{point.x, point.y, point.z});
        nor1_.push_back(Eigen::Vector3d{point.normal_x, point.normal_y, point.normal_z});
        lidar_id_vec1_.push_back(TOP_LIDAR);
    }
    for(auto& point : cloud1_back_->points) {
        pts1_.push_back(Eigen::Vector3d{point.x, point.y, point.z});
        nor1_.push_back(Eigen::Vector3d{point.normal_x, point.normal_y, point.normal_z});
        lidar_id_vec1_.push_back(BACK_LIDAR);
    }
    for(auto& point : cloud2_top_->points) {
        pts2_.push_back(Eigen::Vector3d{point.x, point.y, point.z});
        nor2_.push_back(Eigen::Vector3d{point.normal_x, point.normal_y, point.normal_z});
        lidar_id_vec2_.push_back(TOP_LIDAR);
    }
    for(auto& point : cloud2_back_->points) {
        pts2_.push_back(Eigen::Vector3d{point.x, point.y, point.z});
        nor2_.push_back(Eigen::Vector3d{point.normal_x, point.normal_y, point.normal_z});
        lidar_id_vec2_.push_back(BACK_LIDAR);
    }

    refreshUnifiedPts();
}

// 调用之前必须先更新好move_t和外参
void Frame::refreshUnifiedPts() {
    unified_pts1_ = pts1_;
    unified_nor1_ = nor1_;
    unified_pts2_ = pts2_;
    unified_nor2_ = nor2_;

    Eigen::Isometry3d temp_T = Eigen::Isometry3d::Identity();
    // for 1, only the 2nd half
    temp_T.linear() = back_to_top_extrinsic_.linear();
    temp_T.translation() = back_to_top_extrinsic_.translation();
    transPointsVec(temp_T, unified_pts1_, num1_top_, unified_pts1_.size()-1);
    temp_T.translation() = Eigen::Vector3d(0,0,0);
    transPointsVec(temp_T, unified_nor1_, num1_top_, unified_nor1_.size()-1);
    // for 2, trans the 1st half
    temp_T = extrinsic_.inverse() * move_t_ * extrinsic_;
    transPointsVec(temp_T, unified_pts2_, 0, num2_top_-1);
    temp_T.translation() = Eigen::Vector3d(0,0,0);
    transPointsVec(temp_T, unified_nor2_, 0, num2_top_-1);
    // for 2, trans the 2nd half
    temp_T = extrinsic_.inverse() * move_t_ * extrinsic_ * back_to_top_extrinsic_;
    transPointsVec(temp_T, unified_pts2_, num2_top_, unified_pts2_.size()-1);
    temp_T.translation() = Eigen::Vector3d(0,0,0);
    transPointsVec(temp_T, unified_nor2_, num2_top_, unified_nor2_.size()-1);
}

double Frame::getClosestPoint(
		const Eigen::Vector3d &query_pt,
		size_t &ret_index) {  // query_pt must be in dstFrame
    mutex_.lock();
	if (!index_computed_) {
		index_ptr_ = new MyKdTreeT(
				3 /*dim*/, *this,
				nanoflann::KDTreeSingleIndexAdaptorParams(1 /* max leaf */));
		index_ptr_->buildIndex();
		//        cout << "flann: build index" << endl;
		index_computed_ = true;
	}
    mutex_.unlock();
	// do a knn search
	const size_t num_results = 1;
	// size_t ret_index;
	double out_dist_sqr;
	nanoflann::KNNResultSet<double> resultSet(num_results);
	resultSet.init(&ret_index, &out_dist_sqr);
	
	index_ptr_->findNeighbors(resultSet, &query_pt[0],
	                          nanoflann::SearchParams(32, 0, false));
	
	return out_dist_sqr;
}

void Frame::transPointsSingleThread(const Eigen::Isometry3d& rt, vector<Eigen::Vector3d>& vp, int l, int r) {
    for(int i=l; i<=r; ++i) {
        vp[i] = rt.linear() * vp[i] + rt.translation();
    }
}

void Frame::transPointsVec(const Eigen::Isometry3d& rt, vector<Eigen::Vector3d>& vp, int start, int end) {
    vector<thread> threads;

    if(start<0 || end >= vp.size()) cout << "transPointsVec erroe!" << endl;
    int size = end-start+1;
    for(int i=1; i<=num_threads_; ++i) {
        int l = start + (i-1) * size * 1.0 / num_threads_;
        int r = start + i * size * 1.0 / num_threads_ - 1;
//        cout << "l = " << l << endl;
//        cout << "r = " << r << endl;
        threads.push_back(std::move(std::thread(&Frame::transPointsSingleThread, this, std::ref(rt), std::ref(vp), l, r)));
    }

    for(int i=0; i<threads.size(); ++i) {
        threads[i].join();
    }
}

void Frame::getClosestPointsSingleThread(vector<Eigen::Vector3d>& pts_vec,
        int l, int r,
        vector<double>& dists, double distance_thresh) {
    for(int i=l; i<=r; ++i) {
        size_t idx_min;
        double point_dist_squared;
        point_dist_squared = this->getClosestPoint(pts_vec[i], idx_min);

        double point_dist = 1e10;
        point_dist = sqrt(point_dist_squared);

        if (point_dist < distance_thresh) {
          Correspondence coor;
          if(&pts_vec == &unified_pts2_) {
            coor.local_flag = false;
            coor.first_ = idx_min;
            coor.lidar_id_first = lidar_id_vec1_[coor.first_];
            coor.is_ground_first = fabs(unified_nor1_[coor.first_][2]) > 0.98;
            coor.second_ = i;
            coor.lidar_id_second = lidar_id_vec2_[coor.second_];
            coor.is_ground_second = fabs(unified_nor2_[coor.second_][2]) > 0.98;
            coor.dist_ = point_dist;
            coor.dis = (pts1_[coor.first_].norm() + pts2_[coor.second_].norm()) / 2;
            if(coor.dis < 10) continue;
          }
          else { // local
            coor.local_flag = true;
            coor.first_ = idx_min;
            coor.lidar_id_first = lidar_id_vec1_[coor.first_];
            coor.is_ground_first = fabs(unified_nor1_[coor.first_][2]) > 0.98;
            coor.second_ = i;
            coor.lidar_id_second = lidar_id_vec1_[coor.second_];
            coor.is_ground_second = fabs(unified_nor1_[coor.second_][2]) > 0.98;
            coor.dist_ = point_dist;
            coor.dis = (pts1_[coor.first_].norm() + pts1_[coor.second_].norm()) / 2;
            if(coor.dis < 10) continue;
          }
            mutex_.lock();
            this->corr_vec_.push_back(coor);
            dists.push_back(point_dist);
            mutex_.unlock();
        }
    }
}

void Frame::getClosestPoints(double distance_thresh) {
    corr_vec_.clear();

    std::vector<double> dists;

    vector<thread> threads;

    int size = unified_pts2_.size();
    for(int i=1; i<=num_threads_; ++i) {
        int l = 0 + (i-1) * size * 1.0 / num_threads_;
        int r = 0 + i * size * 1.0 / num_threads_ - 1;

        threads.push_back(std::move(std::thread(&Frame::getClosestPointsSingleThread, this,
                std::ref(unified_pts2_), l, r,
                std::ref(dists), distance_thresh
                )));
    }

    for(int i=0; i<threads.size(); ++i) {
        threads[i].join();
    }

    std::vector<double>::iterator middle = dists.begin() + (dists.size() / 2);
    std::nth_element(dists.begin(), middle, dists.end());

    double nthValue;
    if (dists.size() > 0)
        nthValue = *middle;
    else
        nthValue = 0.5;

    weight_ = nthValue * 1.5;
}

void Frame::getClosestPoints(std::vector<Eigen::Vector3d>& pts_vec, double distance_thresh, int left, int right) {
  std::vector<double> dists;
  vector<thread> threads;
  int size = right - left + 1;

  for(int i=1; i<=num_threads_; ++i) {
    int l = left + (i-1) * size * 1.0 / num_threads_;
    int r = left + i * size * 1.0 / num_threads_ - 1;

    threads.push_back(std::move(std::thread(&Frame::getClosestPointsSingleThread, this,
                                            std::ref(pts_vec), l, r,
                                            std::ref(dists), distance_thresh
    )));
  }

  for(int i=0; i<threads.size(); ++i) {
    threads[i].join();
  }

  std::vector<double>::iterator middle = dists.begin() + (dists.size() / 2);
  std::nth_element(dists.begin(), middle, dists.end());

  double nthValue;
  if (dists.size() > 0)
    nthValue = *middle;
  else
    nthValue = 0.5;

  weight_ = nthValue * 1.5;
}





//void Frame::getClosestPoints(double distance_thresh) {
//	corr_vec_.clear();
//
//    std::vector<double> dists;
//	for(int i=0; i<unified_pts2_.size(); ++i) {
//        size_t idx_min;
//        double point_dist_squared;
//        point_dist_squared = this->getClosestPoint(unified_pts2_[i], idx_min);
//
//        double point_dist = 1e10;
//        point_dist = sqrt(point_dist_squared);
//
//        if (point_dist < distance_thresh) {
//            Correspondence coor;
//            coor.first_ = idx_min;
//            coor.lidar_id_first = lidar_id_vec1_[coor.first_];
//            coor.second_ = i;
//            coor.lidar_id_second = lidar_id_vec2_[coor.second_];
//            coor.dist_ = point_dist;
//            this->corr_vec_.push_back(coor);
//            dists.push_back(point_dist);
//        }
//	}
//
//	std::vector<double>::iterator middle = dists.begin() + (dists.size() / 2);
//	std::nth_element(dists.begin(), middle, dists.end());
//
//	double nthValue;
//	if (dists.size() > 0)
//		nthValue = *middle;
//	else
//		nthValue = 0.5;
//
//	weight_ = nthValue * 1.5;
//}

//void Frame::filterPtsByCorrespondences(double thresh) {
//	index_computed_ = false;
//
//	std::cout << "before filterPtsByCorrespondences : pts1 num is "
//	          << pts1_.size() << endl;
//	std::cout << "before filterPtsByCorrespondences : pts2 num is "
//	          << pts2_.size() << endl;
//	getClosestPoints(thresh);
//	vector<Eigen::Vector3d> pts1_tmp, nor1_tmp;
//	for (size_t i = 0; i < corr_vec_.size(); i++) {
//		pts1_tmp.emplace_back(pts1_.at(corr_vec_[i].first_));
//		nor1_tmp.emplace_back(nor1_.at(corr_vec_[i].first_));
//	}
//	swap(pts1_, pts1_tmp);
//	swap(nor1_, nor1_tmp);
//
//	swap(pts1_, pts2_);
//	swap(nor1_, nor2_);
//	Eigen::Isometry3d move_t_tmp = move_t_;
//	move_t_ = move_t_.inverse();
//
//	index_computed_ = false;
//
//	getClosestPoints(thresh);
//	vector<Eigen::Vector3d> pts2_tmp, nor2_tmp;
//	for (size_t i = 0; i < corr_vec_.size(); i++) {
//		pts2_tmp.emplace_back(pts1_.at(corr_vec_[i].first_));
//		nor2_tmp.emplace_back(nor1_.at(corr_vec_[i].first_));
//	}
//
//	swap(pts1_, pts2_tmp);
//	swap(nor1_, nor2_tmp);
//
//	swap(pts1_, pts2_);
//	swap(nor1_, nor2_);
//
//	move_t_ = move_t_tmp;
//
//	index_computed_ = false;
//
//	std::cout << "after filterPtsByCorrespondences : pts1 num is " << pts1_.size()
//	          << endl;
//	std::cout << "after filterPtsByCorrespondences : pts2 num is " << pts2_.size()
//	          << endl;
//}

void Frame::filterCorrespondences() {
	std::sort(corr_vec_.begin(), corr_vec_.end(), comp);
	
	for (size_t i = 1; i < corr_vec_.size(); i++) {
		if (corr_vec_[i].first_ == corr_vec_[i - 1].first_) {
			if (corr_vec_[i].dist_ > corr_vec_[i - 1].dist_)
				swap(corr_vec_[i], corr_vec_[i - 1]);
		}
	}
	
	std::vector<Correspondence> filtered_corr_vec;
	for (size_t i = 0; i < corr_vec_.size(); i++) {
		if (i == corr_vec_.size() - 1) {
			filtered_corr_vec.push_back(corr_vec_[i]);
		} else if (corr_vec_[i].first_ != corr_vec_[i + 1].first_) {
			filtered_corr_vec.push_back(corr_vec_[i]);
		}
	}
	
	swap(corr_vec_, filtered_corr_vec);
}

void Frame::filterCorrespondencesByNormal() {
	std::vector<Correspondence> filtered_corr_vec;
	for (size_t i = 1; i < corr_vec_.size(); i++) {
		Eigen::Vector3d normal1 = unified_nor1_.at(corr_vec_[i].first_);
		Eigen::Vector3d normal2 = unified_nor2_.at(corr_vec_[i].second_);

		double match = normal1.dot(normal2);
		if (fabs(match) > 0.94) filtered_corr_vec.push_back(corr_vec_[i]);
	}
	//    std::cout << "before normal filter, num of corr is :" <<
	//    corr_vec_.size()
	//    << std::endl;
	swap(corr_vec_, filtered_corr_vec);
	//    std::cout << "after normal filter, num of corr is :" << corr_vec_.size()
	//    << std::endl;
}

namespace {
void cloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, bool downsampling,
                 bool outlierfilter, bool removeNaN) {
	if (downsampling) {
		pcl::VoxelGrid<pcl::PointXYZI> grid;
		
		grid.setLeafSize(0.05, 0.05, 0.05);
		grid.setInputCloud(cloud);
		grid.filter(*cloud);
	}
	
	if (outlierfilter) {
		pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
		sor.setMeanK(50);
		sor.setStddevMulThresh(1.0);
		sor.setInputCloud(cloud);
		sor.filter(*cloud);
	}
	
	if (removeNaN) {
		std::vector<int> indices;
		pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	}
}

PointCloud::Ptr cloudCalculateNormal(
		pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud_in) {
	PointCloud::Ptr cloud_out(new PointCloud);
	pcl::copyPointCloud(*cloud_in, *cloud_out);
	
	pcl::NormalEstimation<pcl::PointXYZI, PointT> norm_est;
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
			new pcl::search::KdTree<pcl::PointXYZI>());
	norm_est.setSearchMethod(tree);
	norm_est.setKSearch(10);
	norm_est.setInputCloud(cloud_in);
	norm_est.compute(*cloud_out);
	
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_out, *cloud_out, indices);
	
	return cloud_out;
}

void cloudFilterCurvature(PointCloud::Ptr &cloud) {
	pcl::ConditionOr<pcl::PointNormal>::Ptr curva_cond(
			new pcl::ConditionOr<pcl::PointNormal>);
	curva_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
			new pcl::FieldComparison<pcl::PointNormal>(
					"curvature", pcl::ComparisonOps::LT, 0.07)));
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(curva_cond);
	condrem.setInputCloud(cloud);
	condrem.filter(*cloud);
	
	//    pcl::RadiusOutlierRemoval<PointT> outrem;
	//    outrem.setInputCloud(cloud);
	//    outrem.setRadiusSearch(0.8);
	//    outrem.setMinNeighborsInRadius(4);
	//    outrem.filter(*cloud);
	PointCloud::Ptr out(new PointCloud);
	
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setMeanK(10);
	sor.setStddevMulThresh(1.0);
	sor.setInputCloud(cloud);
	sor.filter(*cloud);
}
}



PointCloud::Ptr cloudPreProcess(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,
                                bool downsampling, bool outlierfilter,
                                bool removeNaN) {
	cloudFilter(cloud_in, downsampling, outlierfilter, removeNaN);
	
	PointCloud::Ptr cloud_out = cloudCalculateNormal(cloud_in);
	
//	cloudFilterCurvature(cloud_out);
	
	return cloud_out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr transVecToPCL(const vector<Eigen::Vector3d>& vp) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(const auto& p : vp) {
        pcl::PointXYZ pt;
        pt.x = p[0];
        pt.y = p[1];
        pt.z = p[2];
        cloud->points.emplace_back(std::move(pt));
    }
    return cloud;
}
}// namespace