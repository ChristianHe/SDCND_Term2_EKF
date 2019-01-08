#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  //????

  //measurement function - laser 
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement Jacobians - radar
  Hj_ << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  //data shall be first visualized to get a general idea with what, python plot?
/*   if(measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    cout << "meas_pack L: " << measurement_pack.raw_measurements_(0) << " " 
                            << measurement_pack.raw_measurements_(1) << " "
                            << measurement_pack.timestamp_
                            << endl;
  }
  else if(measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    cout << "meas_pack R: " << measurement_pack.raw_measurements_(0) << " " 
                            << measurement_pack.raw_measurements_(1) << " "
                            << measurement_pack.raw_measurements_(2) << " "
                            << measurement_pack.timestamp_
                            << endl;
  } */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    Eigen::VectorXd x_in = VectorXd(4);
    Eigen::MatrixXd P_in = MatrixXd(4,4);
    Eigen::MatrixXd F_in = MatrixXd(4,4);
    Eigen::MatrixXd Q_in = MatrixXd(4,4);
    //P_in will be updated every time predicted, therefore its intial value is unimportant.
    P_in << 100, 0, 0, 0,
            0, 100, 0, 0,
            0, 0, 100, 0,
            0, 0, 0, 100;
    //no delta_t in the first measurement, therefore set to 0.
    F_in << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    Q_in << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;   

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      cout << "Initialize KF with first RADAR data" << endl;
      float ro     = measurement_pack.raw_measurements_(0);
      float theta  = measurement_pack.raw_measurements_(1);
      float px = ro * cos(theta); //x is verical
      float py = ro * sin(theta); //y is horizon
      float vx = 0;
      float vy = 0;
      x_in << px, py, vx, vy; 
      cout << "initial px py: " << px << py << endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      cout << "Initialize KF with first LASER data." << endl;

      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);
      float vx = 0;
      float vy = 0;
      x_in << px, py, vx, vy; 
    }   
    
    ekf_.Init(x_in, P_in, F_in, H_laser_, R_laser_, Q_in);

    // record the timestamp for next time useage.
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  //delta_t is only used in predict process.
  //timestamp unit is 1/1000000 second. 
  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  //cout << "delta_t" << delta_t << endl;
  //prepare the F_ and Q_ with the delta_t
  ekf_.F_ << 1, 0, delta_t, 0,
             0, 1, 0, delta_t,
             0, 0, 1, 0,
             0, 0, 0, 1;

  Eigen::MatrixXd Qv_ = MatrixXd(2, 2);
  Eigen::MatrixXd G_  = MatrixXd(4, 2);
  Eigen::MatrixXd Gt_ = MatrixXd(2, 4);
  Qv_ << 9, 0,
         0, 9;
  G_ <<  delta_t * delta_t / 2, 0,
         0, delta_t * delta_t / 2,
         delta_t, 0,
         0, delta_t;
  Gt_ = G_.transpose(); 

  ekf_.Q_ = G_ * Qv_ * Gt_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    //prepare the z_
    float ro     = measurement_pack.raw_measurements_(0);
    float theta  = measurement_pack.raw_measurements_(1);
    float ro_dot = measurement_pack.raw_measurements_(2);
    Eigen::VectorXd z_ = VectorXd(3, 1);
    z_ << ro, theta, ro_dot; 

    ekf_.UpdateEKF(z_);

  } else {
    // Laser updates
    //prepare the z_
    float px = measurement_pack.raw_measurements_(0);
    float py = measurement_pack.raw_measurements_(1);
    Eigen::VectorXd z_ = VectorXd(2, 1);
    z_ << px, py;

    ekf_.Update(z_);
  }

  //record the timestamp for next use
  previous_timestamp_ = measurement_pack.timestamp_;

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
