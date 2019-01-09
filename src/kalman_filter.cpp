#include <iostream>
#include <math.h>
#include "kalman_filter.h"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, 
                        Eigen::MatrixXd &F_in, Eigen::MatrixXd &Q_in,
                        Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in,
                        Eigen::MatrixXd &EH_in, Eigen::MatrixXd &ER_in){
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;
  EH_ = EH_in;
  ER_ = ER_in;
  pre_x_ = x_;

  //LogFile.open("../log.txt");
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft_ = F_.transpose();
  P_ = F_ * P_ * Ft_ + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y_ = z - H_ * x_;
  MatrixXd Ht_ = H_.transpose();
  MatrixXd S_ = H_ * P_ * Ht_ + R_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ =  P_ * Ht_ * Si_;
  MatrixXd I;
  I = MatrixXd::Identity(4, 4);

  // new state
  x_ = x_ + (K_ * y_);
  P_ = (I - K_ * H_) * P_;

  //preserve the x_ for talor expansion.
  pre_x_ = x_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  Tools tools;
 
  //calculate hx with hx = hu + Hjacobians * (x - u)
  //set u equal to the previous state x.
  VectorXd u_ = pre_x_;
  float u_px = u_(0);
  float u_py = u_(1);
  float u_vx = u_(2);
  float u_vy = u_(3);

  //calculate the hu
  VectorXd hu_ = VectorXd(3, 1);
  // pre-compute a set of terms to avoid repeated calculation
  float c1 = u_px*u_px+u_py*u_py;
  float c2 = sqrt(c1);

  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "UKF - Error - Division by Zero" << endl;
    return;
  }

  float hu1 = c2;
  float hu2 = atan2(u_py, u_px);
  float hu3 = (u_px*u_vx+u_py*u_vy)/c2;
  
  //when cross from -pi to pi, add 2*pi
  if(hu2 < 0 && z(1) > 0)
  {
    hu2 += 2 * 3.1415;
  }

  //when cross from pi to -pi, sub 2*pi
  if(hu2 > 0 && z(1) < 0)
  {
    hu2 -= 2 * 3.1415;
  }

  hu_ << hu1, 
         hu2,
         hu3;

  //calculate the Hjacobians.
  EH_ = tools.CalculateJacobian(x_);

  VectorXd hx_ = hu_ + EH_ * (x_ - u_); 
  VectorXd y_ = z - hx_;
  MatrixXd EHt_ = EH_.transpose();
  MatrixXd S_ = EH_ * P_ * EHt_ + ER_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ =  P_ * EHt_ * Si_;
  MatrixXd I;
  I = MatrixXd::Identity(4, 4);

  /*LogFile << "y1: " << y_(1) << " z1: " << z(1) << " hx1: " << hx_(1) << " hu2: " << hu2 << 
  " u_px: " << u_px << " u_py: " << u_py << " px: " << x_(0) << " py: " << x_(1) << endl;*/
  
  // new state
  x_ = x_ + (K_ * y_);
  P_ = (I - K_ * EH_) * P_;  
  
  //preserve the x_
  pre_x_ = x_;
}
