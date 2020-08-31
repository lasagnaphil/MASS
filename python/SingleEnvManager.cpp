//
// Created by lasagnaphil on 20. 8. 31..
//

#include "SingleEnvManager.h"

SingleEnvManager::
SingleEnvManager(std::string meta_file)
        : env()
{
    dart::math::seedRand();

    env.Initialize(meta_file, false);
}
int
SingleEnvManager::
GetNumState()
{
    return env.GetNumState();
}
int
SingleEnvManager::
GetNumAction()
{
    return env.GetNumAction();
}
int
SingleEnvManager::
GetSimulationHz()
{
    return env.GetSimulationHz();
}
int
SingleEnvManager::
GetControlHz()
{
    return env.GetControlHz();
}
int
SingleEnvManager::
GetNumSteps()
{
    return env.GetNumSteps();
}
bool
SingleEnvManager::
UseMuscle()
{
    return env.GetUseMuscle();
}
void
SingleEnvManager::
Step()
{
    env.Step();
}
void
SingleEnvManager::
Reset(bool RSI)
{
    env.Reset(RSI);
}
bool
SingleEnvManager::
IsEndOfEpisode()
{
    return env.IsEndOfEpisode();
}
py::array_t<float>
SingleEnvManager::
GetState()
{
    return toNumPyArray(env.GetState());
}
void
SingleEnvManager::
SetAction(py::array_t<float> np_array)
{
    env.SetAction(toEigenVector(np_array));
}
double
SingleEnvManager::
GetReward()
{
    return env.GetReward();
}

void
SingleEnvManager::
Steps(int num)
{
    for (int j = 0; j < num; j++) {
        env.Step();
    }
}
void
SingleEnvManager::
StepsAtOnce()
{
    int num = this->GetNumSteps();

    for (int j = 0; j < num; j++) {
        env.Step();
    }
}

py::array_t<float>
SingleEnvManager::
GetMuscleTorques()
{
    return toNumPyArray(env.GetMuscleTorques());
}
py::array_t<float>
SingleEnvManager::
GetDesiredTorques()
{
    return toNumPyArray(env.GetDesiredTorques());
}

void
SingleEnvManager::
SetActivationLevels(py::array_t<float> np_array)
{
    Eigen::VectorXd activations = toEigenVector(np_array);
    env.SetActivationLevels(activations);
}

py::list
SingleEnvManager::
GetMuscleTuples()
{
    auto& tps = env.GetMuscleTuples();
    py::list t;
    for(int j=0;j<tps.size();j++)
    {
        t.append(toNumPyArray(tps[j].JtA));
        t.append(toNumPyArray(tps[j].tau_des));
        t.append(toNumPyArray(tps[j].L));
        t.append(toNumPyArray(tps[j].b));
    }
    return t;
}
