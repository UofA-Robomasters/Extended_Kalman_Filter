#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include "kalman_filter.h"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  bool has_valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
    cerr << usage_instructions << endl;
  } else if (argc == 2) {
    cerr << "Please include an output file.\n" << usage_instructions << endl;
  } else if (argc == 3) {
    has_valid_args = true;
  } else if (argc > 3) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name) {
  if (!in_file.is_open()) {
    cerr << "Cannot open input file: " << in_name << endl;
    exit(EXIT_FAILURE);
  }

  if (!out_file.is_open()) {
    cerr << "Cannot open output file: " << out_name << endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {

  check_arguments(argc, argv);

  string in_file_name_ = argv[1];
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);

  string out_file_name_ = argv[2];
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);

  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  vector<MeasurementPackage> measurement_pack_list;
  vector<GroundTruthPackage> gt_pack_list;

  string line;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  while (getline(in_file_, line)) {

    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long long timestamp;

    // reads first element from the current line
    iss >> sensor_type;
    if (sensor_type.compare("L") == 0) {
      // LASER MEASUREMENT

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float x;
      float y;
      iss >> x;
      iss >> y;
      meas_package.raw_measurements_ << x, y;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    } else if (sensor_type.compare("R") == 0) {
      // RADAR MEASUREMENT

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float phi;
      float ro_dot;
      iss >> ro;
      iss >> phi;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro, phi, ro_dot;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    }

    // read ground truth data to compare later
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;
    gt_package.gt_values_ = VectorXd(4);
    gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
    gt_pack_list.push_back(gt_package);
  }

  // Create a Fusion EKF instance
  //FusionEKF fusionEKF;
  KalmanFilter ekf;

  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  // declare variables
  bool is_initialized;

  long long previous_timestamp;
  Eigen::MatrixXd R_laser;    // laser measurement noise
  Eigen::MatrixXd H_laser;    // measurement function for laser

  float noise_ax;
  float noise_ay;

  is_initialized = false;

  previous_timestamp = 0;

  // initializing matrices
  R_laser = MatrixXd(2, 2);
  H_laser = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser << 0.0225, 0,
             0, 0.0225;

  H_laser << 1, 0, 0, 0,
             0, 1, 0, 0;

  // initialize the kalman filter variables
  ekf.P_ = MatrixXd(4, 4);
  ekf.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

  ekf.F_ = MatrixXd(4, 4);
  ekf.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

  // set measurement noises
  noise_ax = 9;
  noise_ay = 9;

  MeasurementPackage measurement_pack;

  //Call the EKF-based fusion
  size_t N = measurement_pack_list.size();
  for (size_t k = 0; k < N; ++k) {
    // start filtering from the second frame (the speed is unknown in the first
    // frame)
//    fusionEKF.ProcessMeasurement(measurement_pack_list[k]);
    measurement_pack = measurement_pack_list[k];

    /*****************************************************************************
   *  Initialization
   ****************************************************************************/
    if (!is_initialized) {

      // first measurement
//    cout << "EKF: " << endl;
      ekf.x_ = VectorXd(4);

      if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        /**
        Initialize state.
        */

        ekf.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0; // x, y, vx, vy
      }


      previous_timestamp = measurement_pack.timestamp_;

      // done initializing, no need to predict or update
      is_initialized= true;
    }
    else{
      /*****************************************************************************
     *  Prediction
     ****************************************************************************/

      /**
         * Update the state transition matrix F according to the new elapsed time.
          - Time is measured in seconds.
         * Update the process noise covariance matrix.
         * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
       */

      // compute the time elapsed between the current and previous measurements
      float dt = (measurement_pack.timestamp_ - previous_timestamp) / 1000000.0;  //  in seconds
      previous_timestamp = measurement_pack.timestamp_;

      float dt_2 = dt * dt;
      float dt_3 = dt_2 * dt;
      float dt_4 = dt_3 * dt;

      // Modify the F matrix so that the time is integrated
      ekf.F_(0, 2) = dt;
      ekf.F_(1, 3) = dt;

      //set the process covariance matrix Q
      ekf.Q_ = MatrixXd(4, 4);
      ekf.Q_ << dt_4/4*noise_ax,   0,                dt_3/2*noise_ax,  0,
          0,                 dt_4/4*noise_ay,  0,                dt_3/2*noise_ay,
          dt_3/2*noise_ax,   0,                dt_2*noise_ax,    0,
          0,                 dt_3/2*noise_ay,  0,                dt_2*noise_ay;

      ekf.Predict();

      /*****************************************************************************
       *  Update
       ****************************************************************************/

      /**
         * Use the sensor type to perform the update step.
         * Update the state and covariance matrices.
       */

      if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        ekf.H_ = H_laser;
        ekf.R_ = R_laser;
        ekf.Update(measurement_pack.raw_measurements_);
      }
    }



    // output the estimation
    out_file_ << ekf.x_(0) << "\t";
    out_file_ << ekf.x_(1) << "\t";
    out_file_ << ekf.x_(2) << "\t";
    out_file_ << ekf.x_(3) << "\t";

    // output the measurements
    if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
      // output the estimation
      out_file_ << measurement_pack_list[k].raw_measurements_(0) << "\t";
      out_file_ << measurement_pack_list[k].raw_measurements_(1) << "\t";
    }

    // output the ground truth packages
    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(3) << "\n";

    estimations.push_back(ekf.x_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);
  }

  // compute the accuracy (RMSE)
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }

  if (in_file_.is_open()) {
    in_file_.close();
  }

  return 0;
}
