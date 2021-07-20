#include "icp-ceres.h"

#include <math.h>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <ceres/loss_function.h>

#define PRINT_RESIDUALS


#define useLocalParam

namespace jf {

namespace ICP_Ceres {

ceres::Solver::Options getOptions() {
	// Set a few options
	ceres::Solver::Options options;
	// options.use_nonmonotonic_steps = true;
	// options.preconditioner_type = ceres::IDENTITY;
	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = 50;
	
	//    options.preconditioner_type = ceres::SCHUR_JACOBI;
	//    options.linear_solver_type = ceres::DENSE_SCHUR;
	//    options.use_explicit_schur_complement=true;
	//    options.max_num_iterations = 100;
	
	std::cout << "Ceres Solver getOptions()" << endl;
	std::cout << "Ceres preconditioner type: " << options.preconditioner_type
	          << endl;
	std::cout << "Ceres linear algebra type: "
	          << options.sparse_linear_algebra_library_type << endl;
	std::cout << "Ceres linear solver type: " << options.linear_solver_type
	          << endl;
	
	return options;
}

ceres::Solver::Options getOptionsMedium() {
	// Set a few options
	ceres::Solver::Options options;

#ifdef _WIN32
	options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
#else
	// options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#endif  // _WIN32
	
	// If you are solving small to medium sized problems, consider setting
	// Solver::Options::use_explicit_schur_complement to true, it can result in a
	// substantial performance boost.
	options.use_explicit_schur_complement = true;
	options.max_num_iterations = 50;
	
	std::cout << "Ceres Solver getOptionsMedium()" << endl;
	std::cout << "Ceres preconditioner type: " << options.preconditioner_type
	          << endl;
	std::cout << "Ceres linear algebra type: "
	          << options.sparse_linear_algebra_library_type << endl;
	std::cout << "Ceres linear solver type: " << options.linear_solver_type
	          << endl;
	
	return options;
}

void solve(ceres::Problem &problem, bool smallProblem = false) {
	ceres::Solver::Summary summary;
	ceres::Solve(smallProblem ? getOptions() : getOptionsMedium(), &problem,
	             &summary);
	if (!smallProblem) std::cout << "Final report:\n" << summary.FullReport();
}

Eigen::Isometry3d eigenQuaternionToIso(const Eigen::Quaterniond &q,
                                       const Eigen::Vector3d &t) {
	Eigen::Isometry3d poseFinal = Eigen::Isometry3d::Identity();
	poseFinal.linear() = q.toRotationMatrix();
	poseFinal.translation() = t;
	return poseFinal;  //.cast<float>();
}

void ceresOptimizer(std::vector<std::shared_ptr<Frame>> &frames,
                    bool pointToPlane, bool robust, int step) {
	ceres::Problem problem;
	
	// trans ~Extrinsic to Quaternion (to be optimized)
	Eigen::Quaterniond q;
	Eigen::Vector3d t;
	q = Eigen::Quaterniond(Frame::extrinsic_.linear());
	t = Eigen::Vector3d(Frame::extrinsic_.translation());

	Eigen::Quaterniond q_ll;
	Eigen::Vector3d t_ll;
    q_ll = Eigen::Quaterniond(Frame::back_to_top_extrinsic_.linear());
    t_ll = Eigen::Vector3d(Frame::back_to_top_extrinsic_.translation());
	
	cout << "q = " << q.coeffs() << endl;
	cout << "t = " << t << endl;
    cout << "q_ll = " << q_ll.coeffs() << endl;
    cout << "t_ll = " << t_ll << endl;

	double x = t[0];
	double y = t[1];
	double z = t[2];

	std::cout << "ok ceres" << endl;
	// add edges

	for (size_t frame_id = 0; frame_id < frames.size(); frame_id++) {
		auto &frame = *(frames.at(frame_id));
		auto &correpondances = frame.corr_vec_;
		const Eigen::Isometry3d move_t_ = frame.move_t_;
		for (auto corr : correpondances) {
			ceres::CostFunction* cost_function;
			if (step == CALIB_XY_STEP) {
			} else {
        bool diff = false;
				if (pointToPlane) {
				  float weight = 1.0;
            if(corr.local_flag) {
              cost_function = ICPCostFunctions::PointToPlaneErrorLocal::Create(
                frame.pts1_[corr.first_],
                frame.pts1_[corr.second_],
                frame.nor1_[corr.first_], weight);
            }
				    else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
//              weight /= 2;
                        cost_function = ICPCostFunctions::PointToPlaneErrorTopTop::Create(
                                frame.pts1_[corr.first_],
                                frame.pts2_[corr.second_],
                                frame.nor1_[corr.first_],
                                move_t_, weight);
				    }
				    else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == BACK_LIDAR) {
//				                weight *= 3;
                        cost_function = ICPCostFunctions::PointToPlaneErrorTopBack::Create(
                                frame.pts1_[corr.first_],
                                frame.pts2_[corr.second_],
                                frame.nor1_[corr.first_],
                                move_t_, weight);
                        diff = true;
				    } else if(corr.lidar_id_first == BACK_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
//				                weight *= 3;
                        cost_function = ICPCostFunctions::PointToPlaneErrorBackTop::Create(
                                frame.pts1_[corr.first_],
                                frame.pts2_[corr.second_],
                                frame.nor1_[corr.first_],
                                move_t_, weight);
                        diff = true;
				    }else {
                        cost_function = ICPCostFunctions::PointToPlaneErrorBackBack::Create(
                                frame.pts1_[corr.first_],
                                frame.pts2_[corr.second_],
                                frame.nor1_[corr.first_],
                                move_t_, weight);
				    }
				} else {
				}
				ceres::LossFunction *loss = NULL;
				if (robust)
				  //loss = new ceres::SoftLOneLoss(frame.weight_);

				problem.AddResidualBlock(cost_function, loss, q.coeffs().data(), t.data(),
				q_ll.coeffs().data(), t_ll.data());
			}
		}
	}

#ifdef useLocalParam
	eigen_quaternion::EigenQuaternionParameterization
			*quaternion_parameterization =
			new eigen_quaternion::EigenQuaternionParameterization;

	problem.SetParameterization(q.coeffs().data(), quaternion_parameterization);
    problem.SetParameterization(q_ll.coeffs().data(), quaternion_parameterization);
#endif
	if (step == CALIB_XY_STEP) {
//		problem.SetParameterBlockConstant(q.coeffs().data());
//		problem.SetParameterBlockConstant(&z);
	} else {
		problem.SetParameterBlockConstant(t.data());
//    problem.SetParameterBlockConstant(q.coeffs().data());
//    problem.SetParameterBlockConstant(q_ll.coeffs().data());
        problem.SetParameterBlockConstant(t_ll.data());
	}
	
	solve(problem, false);
	
	if (step == CALIB_XY_STEP) t << x, y, z;

#ifdef PRINT_RESIDUALS
  {
    double local_cost=0, top_top_cost = 0, top_back_cost = 0, back_top_cost = 0, back_back_cost = 0;
    int local_num=0, top_top_num=0, top_back_num=0, back_top_num=0, back_back_num=0;
    for (size_t frame_id = 0; frame_id < frames.size(); frame_id++) {
      auto &frame = *(frames.at(frame_id));
      auto &correpondances = frame.corr_vec_;
      const Eigen::Isometry3d move_t_ = frame.move_t_;
      for (auto corr : correpondances) {
        double residual = 0;
        Eigen::Vector3d dst_p, src_p, nor;
        if(corr.local_flag) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts1_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_*src_p - dst_p).dot(nor);
          local_cost += residual*residual;
          ++local_num;
        }
        else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * src_p - dst_p).dot(nor);
          top_top_cost += residual*residual;
          ++top_top_num;
        }
        else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == BACK_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * Frame::back_to_top_extrinsic_ * src_p - dst_p).dot(nor);
          top_back_cost += residual*residual;
          ++top_back_num;
        }
        else if(corr.lidar_id_first == BACK_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_.inverse() * Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * src_p - dst_p).dot(nor);
          back_top_cost += residual*residual;
          ++back_top_num;
        }
        else {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_.inverse() * Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * Frame::back_to_top_extrinsic_ * src_p - dst_p).dot(nor);
          back_back_cost += residual*residual;
          ++back_back_num;
        }
      }
    }
    cout << "--------------------before ceres----------------------------"<<endl;
    cout << "local_cost = " << 0.5 * local_cost << ", num = " << local_num << ", average = " <<  sqrt(local_cost / local_num) << endl;
    cout << "top_top_cost = " << 0.5 * top_top_cost << ", num = " << top_top_num << ", average = " << sqrt(top_top_cost / top_top_num) << endl;
    cout << "top_back_cost = " << 0.5 * top_back_cost << ", num = " << top_back_num << ", average = " << sqrt(top_back_cost / top_back_num) << endl;
    cout << "back_top_cost = " << 0.5 * back_top_cost << ", num = " << back_top_num << ", average = " << sqrt(back_top_cost / back_top_num) << endl;
    cout << "back_back_cost = " << 0.5 * back_back_cost << ", num = " << back_back_num << ", average = " << sqrt(back_back_cost / back_back_num) << endl;
    cout << "--------------------before ceres----------------------------"<<endl;
  };
#endif
	// update camera poses
	Frame::extrinsic_ = eigenQuaternionToIso(q, t);
    Frame::back_to_top_extrinsic_ = eigenQuaternionToIso(q_ll, t_ll);
#ifdef PRINT_RESIDUALS
  {
    double local_cost=0, top_top_cost = 0, top_back_cost = 0, back_top_cost = 0, back_back_cost = 0;
    int local_num=0, top_top_num=0, top_back_num=0, back_top_num=0, back_back_num=0;
    for (size_t frame_id = 0; frame_id < frames.size(); frame_id++) {
      auto &frame = *(frames.at(frame_id));
      auto &correpondances = frame.corr_vec_;
      const Eigen::Isometry3d move_t_ = frame.move_t_;
      for (auto corr : correpondances) {
        double residual = 0;
        Eigen::Vector3d dst_p, src_p, nor;
        if(corr.local_flag) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts1_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_*src_p - dst_p).dot(nor);
          local_cost += residual*residual;
          ++local_num;
        }
        else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * src_p - dst_p).dot(nor);
          top_top_cost += residual*residual;
          ++top_top_num;
        }
        else if(corr.lidar_id_first == TOP_LIDAR && corr.lidar_id_second == BACK_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * Frame::back_to_top_extrinsic_ * src_p - dst_p).dot(nor);
          top_back_cost += residual*residual;
          ++top_back_num;
        }
        else if(corr.lidar_id_first == BACK_LIDAR && corr.lidar_id_second == TOP_LIDAR) {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_.inverse() * Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * src_p - dst_p).dot(nor);
          back_top_cost += residual*residual;
          ++back_top_num;
        }
        else {
          dst_p = frame.pts1_[corr.first_]; src_p = frame.pts2_[corr.second_]; nor = frame.nor1_[corr.first_];
          residual = (Frame::back_to_top_extrinsic_.inverse() * Frame::extrinsic_.inverse() * frame.move_t_ * Frame::extrinsic_ * Frame::back_to_top_extrinsic_ * src_p - dst_p).dot(nor);
          back_back_cost += residual*residual;
          ++back_back_num;
        }
      }
    }
    cout << "--------------------after ceres----------------------------"<<endl;
    cout << "local_cost = " << 0.5 * local_cost << ", num = " << local_num << ", average = " <<  sqrt(local_cost / local_num) << endl;
    cout << "top_top_cost = " << 0.5 * top_top_cost << ", num = " << top_top_num << ", average = " << sqrt(top_top_cost / top_top_num) << endl;
    cout << "top_back_cost = " << 0.5 * top_back_cost << ", num = " << top_back_num << ", average = " << sqrt(top_back_cost / top_back_num) << endl;
    cout << "back_top_cost = " << 0.5 * back_top_cost << ", num = " << back_top_num << ", average = " << sqrt(back_top_cost / back_top_num) << endl;
    cout << "back_back_cost = " << 0.5 * back_back_cost << ", num = " << back_back_num << ", average = " << sqrt(back_back_cost / back_back_num) << endl;
    cout << "--------------------after ceres----------------------------"<<endl;
  };
#endif

	
	std::cout << "After once optimize: lidar-imu\n"
	          << Frame::extrinsic_.matrix() << std::endl;
    std::cout << "After once optimize: lidar-lidar\n"
              << Frame::back_to_top_extrinsic_.matrix() << std::endl;
}
}
}


