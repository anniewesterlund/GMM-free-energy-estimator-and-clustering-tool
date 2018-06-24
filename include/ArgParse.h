/**********************************
*
* 	ArgParse.h
* 	Parsing flags and argument options via command line.
*
*	Annie Westerlund
*	2018, KTH
*
**********************************/

#include <map>
#include <vector>

class ArgParser{

	public:
		ArgParser():n_str_args_{0},n_int_args_{0},n_double_args_{0},n_bool_args_{0},epilog_{""}{};
		ArgParser(std::string epilog):n_str_args_{0},n_int_args_{0},n_double_args_{0},n_bool_args_{0},epilog_{epilog}{};
		
		void add_parameter(std::string flag, std::string label, std::string type);
		void add_parameter(std::string flag, std::string label, std::string type, std::string default_value);
		void add_parameter(std::string flag, std::string label, std::string type, int default_value);
		void add_parameter(std::string flag, std::string label, std::string type, double default_value);
    	void add_parameter(std::string flag, std::string label, std::string type, bool default_value);
    	
    	void parse_arguments(const int argc, const char* argv[]);
    	
    	void get_value(std::string label, std::string &value);
    	void get_value(std::string label, int &value);
    	void get_value(std::string label, double &value);
    	void get_value(std::string label, bool &value);
   		
   	private:
   		
   		void show_documentation();
   		
   		void set_value_str(int index, std::string value);
   		void set_value_int(int index, int value);
   		void set_value_double(int index, double value);
   		void set_value_bool(int index, bool value);
     	
    	std::map<std::string, std::string> value_map_str_;
    	std::map<std::string, int> value_map_int_;
    	std::map<std::string, int> value_map_double_;
    	std::map<std::string, bool> value_map_bool_;
    	
    	std::vector<std::string> label_str_;
    	std::vector<std::string> label_int_;
    	std::vector<std::string> label_double_;
		std::vector<std::string> label_bool_;
    	
   		std::vector<std::string> flag_str_;
   		std::vector<std::string> flag_int_;
   		std::vector<std::string> flag_double_;
   		std::vector<std::string> flag_bool_;
   		   	
    	std::vector<bool> is_next_input_str_;
		std::vector<bool> is_next_input_int_;
		std::vector<bool> is_next_input_double_;
    	
    	int n_str_args_;
    	int n_int_args_;
    	int n_double_args_;
    	int n_bool_args_;
    	
    	std::string epilog_;
};
