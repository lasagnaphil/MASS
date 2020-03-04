#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "NumPyHelper.h"

namespace py = pybind11;

class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	int GetNumState();
	int GetNumAction();
	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);
	py::array_t<float> GetState(int id);
	void SetAction(py::array_t<float> np_array, int id);
	double GetReward(int id);

	void Steps(int num);
	void StepsAtOnce();
	void Resets(bool RSI);
	py::array_t<float> IsEndOfEpisodes();
	py::array_t<float> GetStates();
	void SetActions(py::array_t<float> np_array);
	py::array_t<float> GetRewards();

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	py::array_t<float> GetMuscleTorques();
	py::array_t<float> GetDesiredTorques();
	void SetActivationLevels(py::array_t<float> np_array);
	
	py::list GetMuscleTuples();
private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif