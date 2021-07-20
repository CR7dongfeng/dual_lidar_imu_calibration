#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/local_parameterization.h>
#include <ceres/types.h>
#include <ceres/rotation.h>

#include "frame.h"
#include "eigen_quaternion.h"

using namespace std;

namespace jf {

namespace ICP_Ceres {
const int CALIB_R_STEP = 1;
const int CALIB_XY_STEP = 2;

void ceresOptimizer(std::vector<std::shared_ptr<Frame>> &frames,
                    bool pointToPlane, bool robust, int step = CALIB_R_STEP);
}

namespace ICPCostFunctions {

struct PointToPlaneErrorGlobal {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	const Eigen::Vector3d p_dst;
	const Eigen::Vector3d p_src;
	const Eigen::Vector3d p_nor;
    const int dst_lidar_id_;
    const int src_lidar_id_;
	const Eigen::Isometry3d move_t_;

	PointToPlaneErrorGlobal(const Eigen::Vector3d &dst,
	                        const Eigen::Vector3d &src,
	                        const Eigen::Vector3d &nor,
                            const int dst_lidar_id,
                            const int src_lidar_id,
	                        const Eigen::Isometry3d &move_t_)
			: p_dst(dst), p_src(src), p_nor(nor),
              dst_lidar_id_(dst_lidar_id), src_lidar_id_(src_lidar_id), move_t_(move_t_){}

	// Factory to hide the construction of the CostFunction object from the client
	// code.

	static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
	                                   const Eigen::Vector3d &src,
	                                   const Eigen::Vector3d &nor,
                                       const int dst_lidar_id,
                                       const int src_lidar_id,
	                                   const Eigen::Isometry3d &move_t_) {
		return (new ceres::AutoDiffCostFunction<PointToPlaneErrorGlobal, 1, 4, 3, 4, 3>(
				new PointToPlaneErrorGlobal(dst, src, nor, dst_lidar_id, src_lidar_id, move_t_)));
	}


	template <typename T>
	bool operator() (const T *const extrin_rotation,
	                const T *const extrin_translation,
	                const T *const extrin_back_to_top_rotation,
	                const T *const extrin_back_to_top_translation,
	                T *residuals) const {
		// Make sure the Eigen::Vector world point is using the ceres::Jet type as
		// it's Scalar type
		Eigen::Matrix<T, 3, 1> src;
		src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
		Eigen::Matrix<T, 3, 1> dst;
		dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
		Eigen::Matrix<T, 3, 1> nor;
		nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);
		
		Eigen::Matrix<T, 3, 3> move_r;
		move_r << T(move_t_(0, 0)), T(move_t_(0, 1)), T(move_t_(0, 2)),
				T(move_t_(1, 0)), T(move_t_(1, 1)), T(move_t_(1, 2)), T(move_t_(2, 0)),
				T(move_t_(2, 1)), T(move_t_(2, 2));
		Eigen::Matrix<T, 3, 1> move_t;
		move_t << T(move_t_(0, 3)), T(move_t_(1, 3)), T(move_t_(2, 3));
		
		Eigen::Quaternion<T> q =
				Eigen::Map<const Eigen::Quaternion<T>>(extrin_rotation);
		Eigen::Matrix<T, 3, 1> t =
				Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_translation);

        Eigen::Quaternion<T> q_ll =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
        Eigen::Matrix<T, 3, 1> t_ll =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

        Eigen::Matrix<T, 3, 1> p_src_lidar_top;
        Eigen::Matrix<T, 3, 1> p_src_body;
        Eigen::Matrix<T, 3, 1> p_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> p_dst_body;
        Eigen::Matrix<T, 3, 1> n_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> n_dst_body;

        Eigen::Matrix<T, 3, 1> p_predict;
        if(dst_lidar_id_ == BACK_LIDAR) {
            p_dst_lidar_top = q_ll.toRotationMatrix() * dst + t_ll;
            n_dst_lidar_top = q_ll.toRotationMatrix() * nor;
        } else {
            p_dst_lidar_top = dst;
        }
        if(src_lidar_id_ == BACK_LIDAR) {
            p_src_lidar_top = q_ll.toRotationMatrix() * src + t_ll;
        } else {
            p_src_lidar_top = src;
        }

        p_src_body = q.toRotationMatrix() * p_src_lidar_top + t;
        p_dst_body = q.toRotationMatrix() * p_dst_lidar_top + t;
        n_dst_body = q.toRotationMatrix() * n_dst_lidar_top;

        p_predict = move_r * p_src_body + move_t;
        residuals[0] = (p_predict - p_dst_body).dot(n_dst_body);
		
		return true;
	}
};

struct PointToPlaneErrorTopTop {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const Eigen::Vector3d p_dst;
    const Eigen::Vector3d p_src;
    const Eigen::Vector3d p_nor;
    const Eigen::Isometry3d move_t_;
  const float weight_;

    PointToPlaneErrorTopTop(const Eigen::Vector3d &dst,
                             const Eigen::Vector3d &src,
                             const Eigen::Vector3d &nor,
                             const Eigen::Isometry3d &move_t_,
                            const float weight)
            : p_dst(dst), p_src(src), p_nor(nor), move_t_(move_t_), weight_(weight){}
    // Factory to hide the construction of the CostFunction object from the client
    // code.
    static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
                                       const Eigen::Vector3d &src,
                                       const Eigen::Vector3d &nor,
                                       const Eigen::Isometry3d &move_t_,
                                       const float weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneErrorTopTop, 1, 4, 3, 4, 3>(
                new PointToPlaneErrorTopTop(dst, src, nor, move_t_, weight)));
    }

    template <typename T>
    bool operator() (const T *const extrin_rotation,
                     const T *const extrin_translation,
                     const T *const extrin_back_to_top_rotation,
                     const T *const extrin_back_to_top_translation,
                     T *residuals) const {
        // Make sure the Eigen::Vector world point is using the ceres::Jet type as
        // it's Scalar type
        Eigen::Matrix<T, 3, 1> src;
        src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T, 3, 1> dst;
        dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
        Eigen::Matrix<T, 3, 1> nor;
        nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

        Eigen::Matrix<T, 3, 3> move_r;
        move_r << T(move_t_(0, 0)), T(move_t_(0, 1)), T(move_t_(0, 2)),
                T(move_t_(1, 0)), T(move_t_(1, 1)), T(move_t_(1, 2)), T(move_t_(2, 0)),
                T(move_t_(2, 1)), T(move_t_(2, 2));
        Eigen::Matrix<T, 3, 1> move_t;
        move_t << T(move_t_(0, 3)), T(move_t_(1, 3)), T(move_t_(2, 3));

        Eigen::Quaternion<T> q =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_rotation);
        Eigen::Matrix<T, 3, 1> t =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_translation);

        Eigen::Quaternion<T> q_ll =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
        Eigen::Matrix<T, 3, 1> t_ll =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

        Eigen::Matrix<T, 3, 1> p_src_lidar_top;
        Eigen::Matrix<T, 3, 1> p_src_body;
        Eigen::Matrix<T, 3, 1> p_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> p_dst_body;
        Eigen::Matrix<T, 3, 1> n_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> n_dst_body;

        Eigen::Matrix<T, 3, 1> p_predict;
        p_dst_lidar_top = dst;
        n_dst_lidar_top = nor;

        p_src_lidar_top = src;

        p_src_body = q.toRotationMatrix() * p_src_lidar_top + t;
        p_dst_body = q.toRotationMatrix() * p_dst_lidar_top + t;
        n_dst_body = q.toRotationMatrix() * n_dst_lidar_top;

        p_predict = move_r * p_src_body + move_t;
        residuals[0] = T(weight_) * (p_predict - p_dst_body).dot(n_dst_body);

        return true;
    }
};

struct PointToPlaneErrorTopBack {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const Eigen::Vector3d p_dst;
    const Eigen::Vector3d p_src;
    const Eigen::Vector3d p_nor;
    const Eigen::Isometry3d move_t_;
  const float weight_;

    PointToPlaneErrorTopBack(const Eigen::Vector3d &dst,
                              const Eigen::Vector3d &src,
                              const Eigen::Vector3d &nor,
                              const Eigen::Isometry3d &move_t_,
                             const float weight)
            : p_dst(dst), p_src(src), p_nor(nor), move_t_(move_t_), weight_(weight){}
    // Factory to hide the construction of the CostFunction object from the client
    // code.
    static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
                                       const Eigen::Vector3d &src,
                                       const Eigen::Vector3d &nor,
                                       const Eigen::Isometry3d &move_t_,
                                       const float weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneErrorTopBack, 1, 4, 3, 4, 3>(
                new PointToPlaneErrorTopBack(dst, src, nor, move_t_, weight)));
    }

    template <typename T>
    bool operator() (const T *const extrin_rotation,
                     const T *const extrin_translation,
                     const T *const extrin_back_to_top_rotation,
                     const T *const extrin_back_to_top_translation,
                     T *residuals) const {
        // Make sure the Eigen::Vector world point is using the ceres::Jet type as
        // it's Scalar type
        Eigen::Matrix<T, 3, 1> src;
        src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T, 3, 1> dst;
        dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
        Eigen::Matrix<T, 3, 1> nor;
        nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

        Eigen::Matrix<T, 3, 3> move_r;
        move_r << T(move_t_(0, 0)), T(move_t_(0, 1)), T(move_t_(0, 2)),
                T(move_t_(1, 0)), T(move_t_(1, 1)), T(move_t_(1, 2)), T(move_t_(2, 0)),
                T(move_t_(2, 1)), T(move_t_(2, 2));
        Eigen::Matrix<T, 3, 1> move_t;
        move_t << T(move_t_(0, 3)), T(move_t_(1, 3)), T(move_t_(2, 3));

        Eigen::Quaternion<T> q =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_rotation);
        Eigen::Matrix<T, 3, 1> t =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_translation);

        Eigen::Quaternion<T> q_ll =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
        Eigen::Matrix<T, 3, 1> t_ll =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

        Eigen::Matrix<T, 3, 1> p_src_lidar_top;
        Eigen::Matrix<T, 3, 1> p_src_body;
        Eigen::Matrix<T, 3, 1> p_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> p_dst_body;
        Eigen::Matrix<T, 3, 1> n_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> n_dst_body;

        Eigen::Matrix<T, 3, 1> p_predict;
        p_dst_lidar_top = dst;
        n_dst_lidar_top = nor;

        p_src_lidar_top = q_ll.toRotationMatrix() * src + t_ll;


        p_src_body = q.toRotationMatrix() * p_src_lidar_top + t;
        p_dst_body = q.toRotationMatrix() * p_dst_lidar_top + t;
        n_dst_body = q.toRotationMatrix() * n_dst_lidar_top;

        p_predict = move_r * p_src_body + move_t;
        residuals[0] = T(weight_) * (p_predict - p_dst_body).dot(n_dst_body);

        return true;
    }
};

struct PointToPlaneErrorBackTop {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const Eigen::Vector3d p_dst;
    const Eigen::Vector3d p_src;
    const Eigen::Vector3d p_nor;
    const Eigen::Isometry3d move_t_;
    const float weight_;

    PointToPlaneErrorBackTop(const Eigen::Vector3d &dst,
                              const Eigen::Vector3d &src,
                              const Eigen::Vector3d &nor,
                              const Eigen::Isometry3d &move_t_,
                              const float weight)
            : p_dst(dst), p_src(src), p_nor(nor), move_t_(move_t_), weight_(weight){}
    // Factory to hide the construction of the CostFunction object from the client
    // code.
    static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
                                       const Eigen::Vector3d &src,
                                       const Eigen::Vector3d &nor,
                                       const Eigen::Isometry3d &move_t_,
                                       const float weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneErrorBackTop, 1, 4, 3, 4, 3>(
                new PointToPlaneErrorBackTop(dst, src, nor, move_t_, weight)));
    }

    template <typename T>
    bool operator() (const T *const extrin_rotation,
                     const T *const extrin_translation,
                     const T *const extrin_back_to_top_rotation,
                     const T *const extrin_back_to_top_translation,
                     T *residuals) const {
        // Make sure the Eigen::Vector world point is using the ceres::Jet type as
        // it's Scalar type
        Eigen::Matrix<T, 3, 1> src;
        src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T, 3, 1> dst;
        dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
        Eigen::Matrix<T, 3, 1> nor;
        nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

        Eigen::Matrix<T, 3, 3> move_r;
        move_r << T(move_t_(0, 0)), T(move_t_(0, 1)), T(move_t_(0, 2)),
                T(move_t_(1, 0)), T(move_t_(1, 1)), T(move_t_(1, 2)), T(move_t_(2, 0)),
                T(move_t_(2, 1)), T(move_t_(2, 2));
        Eigen::Matrix<T, 3, 1> move_t;
        move_t << T(move_t_(0, 3)), T(move_t_(1, 3)), T(move_t_(2, 3));

        Eigen::Quaternion<T> q =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_rotation);
        Eigen::Matrix<T, 3, 1> t =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_translation);

        Eigen::Quaternion<T> q_ll =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
        Eigen::Matrix<T, 3, 1> t_ll =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

        Eigen::Matrix<T, 3, 1> p_src_lidar_top;
        Eigen::Matrix<T, 3, 1> p_src_body;
        Eigen::Matrix<T, 3, 1> p_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> p_dst_body;
        Eigen::Matrix<T, 3, 1> n_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> n_dst_body;

        Eigen::Matrix<T, 3, 1> p_predict;
        p_dst_lidar_top = q_ll.toRotationMatrix() * dst + t_ll;
        n_dst_lidar_top = q_ll.toRotationMatrix() * nor;

        p_src_lidar_top = src;

        p_src_body = q.toRotationMatrix() * p_src_lidar_top + t;
        p_dst_body = q.toRotationMatrix() * p_dst_lidar_top + t;
        n_dst_body = q.toRotationMatrix() * n_dst_lidar_top;

        p_predict = move_r * p_src_body + move_t;
        residuals[0] = T(weight_) * (p_predict - p_dst_body).dot(n_dst_body);

        return true;
    }
};

struct PointToPlaneErrorBackBack {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const Eigen::Vector3d p_dst;
    const Eigen::Vector3d p_src;
    const Eigen::Vector3d p_nor;
    const Eigen::Isometry3d move_t_;
  const float weight_;

    PointToPlaneErrorBackBack(const Eigen::Vector3d &dst,
                            const Eigen::Vector3d &src,
                            const Eigen::Vector3d &nor,
                            const Eigen::Isometry3d &move_t_,
                              const float weight)
            : p_dst(dst), p_src(src), p_nor(nor), move_t_(move_t_), weight_(weight){}
    // Factory to hide the construction of the CostFunction object from the client
    // code.
    static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
                                       const Eigen::Vector3d &src,
                                       const Eigen::Vector3d &nor,
                                       const Eigen::Isometry3d &move_t_,
                                       const float weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneErrorBackBack, 1, 4, 3, 4, 3>(
                new PointToPlaneErrorBackBack(dst, src, nor, move_t_, weight)));
    }

    template <typename T>
    bool operator() (const T *const extrin_rotation,
                     const T *const extrin_translation,
                     const T *const extrin_back_to_top_rotation,
                     const T *const extrin_back_to_top_translation,
                     T *residuals) const {
        // Make sure the Eigen::Vector world point is using the ceres::Jet type as
        // it's Scalar type
        Eigen::Matrix<T, 3, 1> src;
        src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
        Eigen::Matrix<T, 3, 1> dst;
        dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
        Eigen::Matrix<T, 3, 1> nor;
        nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

        Eigen::Matrix<T, 3, 3> move_r;
        move_r << T(move_t_(0, 0)), T(move_t_(0, 1)), T(move_t_(0, 2)),
                T(move_t_(1, 0)), T(move_t_(1, 1)), T(move_t_(1, 2)), T(move_t_(2, 0)),
                T(move_t_(2, 1)), T(move_t_(2, 2));
        Eigen::Matrix<T, 3, 1> move_t;
        move_t << T(move_t_(0, 3)), T(move_t_(1, 3)), T(move_t_(2, 3));

        Eigen::Quaternion<T> q =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_rotation);
        Eigen::Matrix<T, 3, 1> t =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_translation);

        Eigen::Quaternion<T> q_ll =
                Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
        Eigen::Matrix<T, 3, 1> t_ll =
                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

        Eigen::Matrix<T, 3, 1> p_src_lidar_top;
        Eigen::Matrix<T, 3, 1> p_src_body;
        Eigen::Matrix<T, 3, 1> p_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> p_dst_body;
        Eigen::Matrix<T, 3, 1> n_dst_lidar_top;
        Eigen::Matrix<T, 3, 1> n_dst_body;

        Eigen::Matrix<T, 3, 1> p_predict;
        p_dst_lidar_top = q_ll.toRotationMatrix() * dst + t_ll;
        n_dst_lidar_top = q_ll.toRotationMatrix() * nor;

        p_src_lidar_top = q_ll.toRotationMatrix() * src + t_ll;

        p_src_body = q.toRotationMatrix() * p_src_lidar_top + t;
        p_dst_body = q.toRotationMatrix() * p_dst_lidar_top + t;
        n_dst_body = q.toRotationMatrix() * n_dst_lidar_top;

        p_predict = move_r * p_src_body + move_t;
        residuals[0] = T(weight_) * (p_predict - p_dst_body).dot(n_dst_body);

        return true;
    }
};

struct PointToPlaneErrorLocal {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const Eigen::Vector3d p_dst;
  const Eigen::Vector3d p_src;
  const Eigen::Vector3d p_nor;
  const float weight_;

  PointToPlaneErrorLocal(const Eigen::Vector3d &dst,
                           const Eigen::Vector3d &src,
                           const Eigen::Vector3d &nor,
                           const float weight)
    : p_dst(dst), p_src(src), p_nor(nor), weight_(weight){}
  // Factory to hide the construction of the CostFunction object from the client
  // code.
  static ceres::CostFunction *Create(const Eigen::Vector3d &dst,
                                     const Eigen::Vector3d &src,
                                     const Eigen::Vector3d &nor,
                                     const float weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<PointToPlaneErrorLocal, 1, 4, 3, 4, 3>(
      new PointToPlaneErrorLocal(dst, src, nor, weight)));
  }

  template <typename T>
  bool operator() (const T *const extrin_rotation,
                   const T *const extrin_translation,
                    const T *const extrin_back_to_top_rotation,
                   const T *const extrin_back_to_top_translation,
                   T *residuals) const {
    // Make sure the Eigen::Vector world point is using the ceres::Jet type as
    // it's Scalar type
    Eigen::Matrix<T, 3, 1> src;
    src << T(p_src[0]), T(p_src[1]), T(p_src[2]);
    Eigen::Matrix<T, 3, 1> dst;
    dst << T(p_dst[0]), T(p_dst[1]), T(p_dst[2]);
    Eigen::Matrix<T, 3, 1> nor;
    nor << T(p_nor[0]), T(p_nor[1]), T(p_nor[2]);

    Eigen::Quaternion<T> q_ll =
      Eigen::Map<const Eigen::Quaternion<T>>(extrin_back_to_top_rotation);
    Eigen::Matrix<T, 3, 1> t_ll =
      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(extrin_back_to_top_translation);

    Eigen::Matrix<T, 3, 1> p_predict;
    p_predict = q_ll.toRotationMatrix() * src + t_ll;
    residuals[0] = T(weight_) * (p_predict - dst).dot(nor);

    return true;
  }
};

} // namespace
} // namespace jf
