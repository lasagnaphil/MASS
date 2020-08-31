//
// Created by lasagnaphil on 20. 8. 31..
//

#ifndef MSS_SINGLEENVMANAGER_H
#define MSS_SINGLEENVMANAGER_H

#include "Environment.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "NumPyHelper.h"

namespace py = pybind11;

class SingleEnvManager
{
public:
    SingleEnvManager(std::string meta_file);

    int GetNumState();
    int GetNumAction();
    int GetSimulationHz();
    int GetControlHz();
    int GetNumSteps();
    bool UseMuscle();

    void Step();
    void Reset(bool RSI);
    bool IsEndOfEpisode();
    py::array_t<float> GetState();
    void SetAction(py::array_t<float> np_array);
    double GetReward();

    void Steps(int num);
    void StepsAtOnce();

    //For Muscle Transitions
    int GetNumTotalMuscleRelatedDofs(){return env.GetNumTotalRelatedDofs();};
    int GetNumMuscles(){return env.GetCharacter()->GetMuscles().size();}
    py::array_t<float> GetMuscleTorques();
    py::array_t<float> GetDesiredTorques();
    void SetActivationLevels(py::array_t<float> np_array);

    py::list GetMuscleTuples();
private:
    MASS::Environment env;

    int mNumEnvs;
};

#endif //MSS_SINGLEENVMANAGER_H
