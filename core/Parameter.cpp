#include "Parameter.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <tinyxml2.h>
#include "Character.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

void Parameter::loadParameter(const std::string& fileName, const std::string& configName) {
    py::scoped_interpreter guard{};

    py::dict globals;
    py::eval_file(fileName, globals);

    py::dict all_configs = globals["CONFIG"].cast<py::dict>();
    py::dict config = all_configs[configName.c_str()]["env_config"];

    loadParameter(config);
}

void Parameter::loadParameter(const py::dict& config) {
#define PARAMASSIGN(TYPE,NAME) if (config.contains(TOSTRING(NAME))) { \
    try { \
        NAME = config[TOSTRING(NAME)].cast<TYPE>(); \
    } catch (const std::exception& e) { \
        std::cerr << "Config error: failed to read field " << TOSTRING(NAME) << '!' << std::endl; \
        std::cerr << "Details: " << e.what() << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
    PARAM(PARAMASSIGN)
}

namespace Parameter{
#define PARAMDECLARATION(TYPE,NAME) TYPE NAME;
	PARAM(PARAMDECLARATION)
	
	int numIteration = 10000;
}
