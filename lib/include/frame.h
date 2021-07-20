#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "nanoflann.hpp"

#include <thread>
#include <mutex>

#define TOP_LIDAR 0
#define BACK_LIDAR 1

using namespace std;

namespace jf{
// point cloud type
typedef pcl::PointNormal PointT;
typedef pcl::PointCloud<PointT> PointCloud;
// corres : 2 indexex & dist
struct Correspondence {
	int first_;
	int lidar_id_first;
	int second_;
	int lidar_id_second;
	bool is_ground_first;
	bool is_ground_second;
	bool local_flag;
	double dist_;
	float dis;
};

/*
 * class Frame
 * 用于标定的基本数据结构
 */
class Frame {
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Frame() : cloud1_top_(new PointCloud), cloud2_top_(new PointCloud),
	            cloud1_back_(new PointCloud), cloud2_back_(new PointCloud) {}
	
	// 1. load from pcd && load move_t_
//	bool loadCloud(const string &filename, int whichCLD);
	
//	void loadMoveT(const string &filename);
	
	// 2. calculate normal
//	void calculateNormal();
	
	// 3. filter + removeNaN
//	void filter(bool downsampling = true, bool outlier_filter = true,
//	            bool removeNaN = true);
	
	// 3+ filter by curvature
//	void filterCurvature();
	
	// 4. copy to Eigen pts
	void copyToEigen();

	void refreshUnifiedPts();
	
	// 5. calculate closest-points correspondences
	void getClosestPoints(double distance_thresh);

	void getClosestPoints(std::vector<Eigen::Vector3d>& pts_vec, double distance_thresh, int left, int right);
	
//	void filterPtsByCorrespondences(double thresh);
	
	void filterCorrespondences();
	
	void filterCorrespondencesByNormal();
	
	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return unified_pts1_.size(); }
	
	// Returns the distance between the vector "p1[0:size-1]" and the data point
	// with index "idx_p2" stored in the class:
	inline double kdtree_distance(const double *p2, const size_t idx_p1,
	                              size_t /*size*/) const {
		const double d0 = p2[0] - unified_pts1_[idx_p1].x();
		const double d1 = p2[1] - unified_pts1_[idx_p1].y();
		const double d2 = p2[2] - unified_pts1_[idx_p1].z();
		const double square_d = d0 * d0 + d1 * d1 + d2 * d2;
		if (square_d > 0.4 * 0.4)
			return square_d;
		else {
			Eigen::Vector3d d(d0, d1, d2);
			double point_to_plane = d.dot(unified_nor1_[idx_p1]);
			return point_to_plane * point_to_plane;
		}
	}
	
	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate
	// value, the
	//  "if/else's" are actually solved at compile time.
	inline double kdtree_get_pt(const size_t idx, int dim) const {
		if (dim == 0)
			return unified_pts1_[idx].x();
		else if (dim == 1)
			return unified_pts1_[idx].y();
		else
			return unified_pts1_[idx].z();
	}
	
	// Optional bounding-box computation: return false to default to a standard
	// bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in
	//   "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3
	//   for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX & /*bb*/) const {
		return false;
	}
 
 private:
	// calculate the closest-point of given point
	double getClosestPoint(const Eigen::Vector3d &query_pt, size_t &ret_index);
	
	static bool comp(const Correspondence &c1, const Correspondence &c2) {
		return (c1.first_ < c2.first_);
	}

	void transPointsVec(const Eigen::Isometry3d& rt, vector<Eigen::Vector3d>& vp, int start, int end);

    void transPointsSingleThread(const Eigen::Isometry3d& rt, vector<Eigen::Vector3d>& vp, int l, int r);

    void getClosestPointsSingleThread(vector<Eigen::Vector3d>& pts_vec,
                                             int l, int r,
                                             vector<double>& dists, double distance_thresh);
 
 public:
	PointCloud::Ptr cloud1_top_;
	PointCloud::Ptr cloud1_back_;
	PointCloud::Ptr cloud2_top_;
	PointCloud::Ptr cloud2_back_;

	int num1_top_;
	int num1_back_;
	int num2_top_;
	int num2_back_;

    vector<Eigen::Vector3d> pts1_;
    vector<Eigen::Vector3d> pts2_;
    vector<Eigen::Vector3d> nor1_;
    vector<Eigen::Vector3d> nor2_;

    vector<int> lidar_id_vec1_;
    vector<int> lidar_id_vec2_;

    vector<Eigen::Vector3d> unified_pts1_;
    vector<Eigen::Vector3d> unified_nor1_;
    vector<Eigen::Vector3d> unified_pts2_;
    vector<Eigen::Vector3d> unified_nor2_;

    Eigen::Isometry3d move_t_;  // the relative imu-movement between 2 poses

    float weight_;
    vector<Correspondence> corr_vec_;

    static Eigen::Isometry3d extrinsic_;
    static Eigen::Isometry3d back_to_top_extrinsic_;

//	vector<Eigen::Vector3d> pts1_;
//	vector<Eigen::Vector3d> pts2_;
//	vector<Eigen::Vector3d> nor1_;
//	vector<Eigen::Vector3d> nor2_;
//	Eigen::Isometry3d move_t_;  // the relative imu-movement between 2 poses
//	float weight_;
//	vector<Correspondence> corr_vec_;
//
//	static Eigen::Isometry3d extrinsic_;
	void resetKDTree() {index_computed_ = false;
		if(index_ptr_ != nullptr) {
			delete(index_ptr_);
			index_ptr_ = nullptr;
		}
	}
 private:
	bool index_computed_ = false;
	
	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<double, Frame>, Frame, 3 /* dim */
	>
		MyKdTreeT;
	
	MyKdTreeT *index_ptr_ = nullptr;

	mutex mutex_;

	int num_threads_ = 10;
};

PointCloud::Ptr cloudPreProcess(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,
                                bool downsampling = true,
                                bool outlierfilter = true,
                                bool removeNaN = true);

pcl::PointCloud<pcl::PointXYZ>::Ptr transVecToPCL(const vector<Eigen::Vector3d>& vp);
}
