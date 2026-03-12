EKF-SLAM from Scratch (Python)
A modular implementation of Extended Kalman Filter (EKF) SLAM for a mobile robot equipped with range-bearing sensors. This project demonstrates the ability to estimate a robot's trajectory while simultaneously building a map of environmental landmarks using non-linear state estimation.

🚀 Overview
Standard Kalman Filters struggle with the non-linearities of robotics (turning, trigonometry, etc.). This project implements an Extended Kalman Filter to linearize these movements using Jacobian matrices, allowing for stable localization and mapping in a simulated 2D environment.

Key Features
1. Non-linear State Estimation: Uses Taylor series expansion to linearize motion and measurement models.
2. Sensor Fusion: Combines noisy wheel odometry with range/bearing landmark observations.
3. Feature-Based Mapping: Dynamically adds new landmarks to the state vector and covariance matrix.
4. Data Association: (Optional: Mention if you used Mahalanobis distance or Nearest Neighbor) to track known landmarks.
5. Real-time Visualization: Matplotlib-based dashboard showing the robot’s path, landmark estimates, and uncertainty ellipses.

🛠️ The Pipeline
Prediction Step: Forecasts the robot's next pose based on velocity commands (Control Input).
Linearization: Calculates Jacobians to propagate uncertainty.
Correction Step: Updates the robot pose and landmark positions based on the difference between expected and actual sensor readings.
State Augmentation: If a new landmark is seen, the state vector expands to include the new coordinates.

💻 Installation & Usage
bash
# Clone the repo
git clone https://github.com

# Install dependencies
pip install numpy matplotlib

# Run the simulation
python main.py

📊 Results
The simulation demonstrates how the uncertainty ellipses (covariance) grow during movement and "shrink" when the robot observes a known landmark, showcasing the EKF's ability to correct drift.
