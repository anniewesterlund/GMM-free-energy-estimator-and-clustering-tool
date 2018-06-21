#include "ArgParse.h"

/**********************************
*
* 	ArgParse.cpp
* 	Parsing flags and argument options via command line.
*
*	Annie Westerlund
*	2018, KTH
*
**********************************/

void ArgParser::add_parameter(std::string flag, std::string label, std::string type){
	// Add an argument parameter. 
	// Separate types = {"str","int","bool"}.
	
	if (type.compare("str")==0){
		
		ArgParser::flag_str_.push_back(flag);
		ArgParser::label_str_.push_back(label);	
		ArgParser::value_map_str_[label] = "";
		ArgParser::is_next_input_str_.push_back(false);
		ArgParser::n_str_args_++;
		
	}else if (type.compare("int")==0){
		
		ArgParser::flag_int_.push_back(flag);
		ArgParser::label_int_.push_back(label);
		ArgParser::value_map_int_[label] = 0;
		ArgParser::is_next_input_int_.push_back(false);
		ArgParser::n_int_args_++;
		
	}else if (type.compare("bool")==0){
		
		ArgParser::flag_bool_.push_back(flag);
		ArgParser::label_bool_.push_back(label);
		ArgParser::value_map_bool_[label] = false;
		ArgParser::n_bool_args_++;
		
	}
}

void ArgParser::add_parameter(std::string flag, std::string label, std::string type, std::string default_value){
	// Add an argument parameter. 
	// Separate types = {"str","int","bool"}.
		ArgParser::flag_str_.push_back(flag);
		ArgParser::label_str_.push_back(label);	
		ArgParser::value_map_str_[label] = default_value;
		ArgParser::is_next_input_str_.push_back(false);
		ArgParser::n_str_args_++;
}

void ArgParser::add_parameter(std::string flag, std::string label, std::string type, int default_value){
	// Add an argument parameter. 
	// Separate types = {"str","int","bool"}.
		ArgParser::flag_int_.push_back(flag);
		ArgParser::label_int_.push_back(label);
		ArgParser::value_map_int_[label] = default_value;
		ArgParser::is_next_input_int_.push_back(false);
		ArgParser::n_int_args_++;
}

void ArgParser::add_parameter(std::string flag, std::string label, std::string type, bool default_value){
	// Add an argument parameter. 
	// Separate types = {"str","int","bool"}.
		ArgParser::flag_bool_.push_back(flag);
		ArgParser::label_bool_.push_back(label);
		ArgParser::value_map_bool_[label] = default_value;
		ArgParser::n_bool_args_++;
}

void ArgParser::set_value_int(int index, int value){
	// Set an int value
	std::string label = ArgParser::label_int_[index];
	ArgParser::value_map_int_[label] = value;
}

void ArgParser::set_value_str(int index, std::string value){
	// Set a str value
	std::string label = ArgParser::label_str_[index];
	ArgParser::value_map_str_[label] = value;
}

void ArgParser::set_value_bool(int index, bool value){
	// Set a bool value
	std::string label = ArgParser::label_bool_[index];
	ArgParser::value_map_bool_[label] = value;
}

void ArgParser::show_documentation(){
	// Print the explanation
	std::clog << ArgParser::epilog_ << std::endl;
}

void ArgParser::parse_arguments(const int argc, const char* argv[]){
	
	std::string input_string;
	bool check_flag = true;
	bool is_file_name = false;
	bool is_dimension = false;
	
	for(int i = 0; i < argc; i++){
		
		input_string = argv[i];
		
		// Update values from input strings
		for(int j = 0; j < ArgParser::n_str_args_; j++){
			if (ArgParser::is_next_input_str_[j]){
				check_flag = false;
				ArgParser::is_next_input_str_[j] = false;
				ArgParser::set_value_str(j, input_string);
				continue;
			}
		}	
		
		// Update values from input ints
		for(int j = 0; j < ArgParser::n_int_args_; j++){
			if (ArgParser::is_next_input_int_[j]){
				check_flag = false;
				ArgParser::is_next_input_int_[j] = false;
				ArgParser::set_value_int(j, std::atoi(argv[i]));
				continue;
			}
		}		
		
		if (check_flag){
			
			// Check for help flag
			if(input_string.compare("-h")==0){
				ArgParser::show_documentation();
				exit(1);
				break;
			}
			
			// Check for string inputs
			for(int j = 0; j < ArgParser::n_str_args_; j++){
				if (input_string.compare(ArgParser::flag_str_[j])==0){
					ArgParser::is_next_input_str_[j] = true;
					continue;
				}
			}	
			
			// Check for int inputs
			for(int j = 0; j < ArgParser::n_int_args_; j++){
				if (input_string.compare(ArgParser::flag_int_[j])==0){
					ArgParser::is_next_input_int_[j] = true;
					continue;
				}
			}
						
			// Check for bool inputs
			for(int j = 0; j < ArgParser::n_bool_args_; j++){
				if (input_string.compare(ArgParser::flag_bool_[j])==0){
					ArgParser::set_value_bool(j, true);
					continue;
				}
			}	
		}
		
		if (!check_flag){
			check_flag = true;
		}
		
	}
}

void ArgParser::get_value(std::string label, std::string &value){
	// Return the value corresponding to the label
	value = ArgParser::value_map_str_[label];
}

void ArgParser::get_value(std::string label, int &value){
	// Return the value corresponding to the label
	value = ArgParser::value_map_int_[label];
}

void  ArgParser::get_value(std::string label, bool &value){
	// Return the value corresponding to the label
	value = ArgParser::value_map_bool_[label];
}