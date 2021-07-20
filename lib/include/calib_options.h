#pragma once

namespace jf {

// calibration parameters
struct CalibrationOptions {
  bool first_time = true;
	int max_iterations = 40;
	double max_correspondence_distance_upper =
			0.4;  // max correspondence distance (max value)
	double max_correspondence_distance_lower =
			0.3;  // min correspondence distance (min value)
	double distance_step =
			0.05;  // the value correspondence distance decrease each time
	int delay_iterations = 3;  // the correspondence distance keeps in n cycles
	int calib_stage = 1;       // calib_stage 1 or 2
	
	double min_move_distance = 0.1;  // the vehicle minimum move distance
	
	double rotation_epsilon = 1e-14;     // inner-loop, rotation epsilon
	double translation_epsilon = 1e-10;  // inner-loop, translation epsilon
	
	double outer_rotation_epsilon = 1e-12;    // outer-loop, rotation epsilon
	double outer_translation_epsilon = 1e-4;  // outer-loop, translation epsilon
};
}
