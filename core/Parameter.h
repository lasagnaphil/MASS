#ifndef MMPARAMETER
#define MMPARAMETER

#include<vector>
#include<string>
#include<set>
#include<map>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using mapstringint = std::map<std::string, int>;
using mapstringstring = std::map<std::string, std::string>;

#define PARAM(MACRO) \
    MACRO(std::string, mass_home) \
	/* Skeleton file */ \
	MACRO(std::string, human) \
	MACRO(std::string, obj) \
	MACRO(double, jointDamping) \
	MACRO(std::vector<std::string>, foot) \
	MACRO(double, footOffset) \
	MACRO(mapstringint, axis) \
	/* Ground file */ \
	MACRO(std::string, ground) \
	/* Muscle file */ \
	MACRO(bool, useMuscle) \
	MACRO(std::string, muscle) \
	/* SimpleMotion file */ \
	MACRO(std::string, simpleMotion) \
	MACRO(std::string, jointMap) \
	/* SPD */ \
	MACRO(double, mKp) \
	/* frame */ \
	MACRO(int, controlHz) \
	MACRO(int, simulationHz) \
	/* motion database list */ \
	MACRO(std::vector<MotionInfo>, motionDataFiles) \
	/* reward components */ \
	MACRO(std::vector<std::string>, endEffectors) \
	MACRO(double, w_q) \
	MACRO(double, w_v) \
	MACRO(double, w_ee) \
	MACRO(double, w_com) \
	/* environment early termination */ \
	MACRO(double, rootHeightLowerLimit) \
	/* environment time limit */ \
	MACRO(double, timeLimit) \
	/* Adaptive sampling */ \
	MACRO(bool, useAdaptiveSampling) \
	MACRO(mapstringstring, symmetry) \
	MACRO(std::vector<std::string>, modifyingBodyNode) \
	MACRO(std::vector<double>, minSkeletonLength) \
	MACRO(std::vector<double>, maxSkeletonLength) \

namespace Parameter{
	using MotionInfo = std::pair<std::string, bool>;
	void loadParameter(const std::string& fileName, const std::string& configName);
    void loadParameter(const py::dict& config);

#define PARAMHEADER(TYPE,NAME) extern TYPE NAME;
	PARAM(PARAMHEADER)

	// temporary parameter
	extern int numIteration;
};

#endif
