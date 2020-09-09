//
// Created by lasagnaphil on 20. 8. 31..
//

#include "SingleEnvManager.h"

#ifdef COMPILE_MULTI_ENV_MANAGER
#include "EnvManager.h"
#endif

PYBIND11_MODULE(pymss, m) {
    py::class_<SingleEnvManager>(m, "SingleEnvManager")
            .def(py::init<const py::dict&>())
            .def("GetNumState", &SingleEnvManager::GetNumState)
            .def("GetNumAction", &SingleEnvManager::GetNumAction)
            .def("GetSimulationHz", &SingleEnvManager::GetSimulationHz)
            .def("GetControlHz", &SingleEnvManager::GetControlHz)
            .def("GetNumSteps", &SingleEnvManager::GetNumSteps)
            .def("UseMuscle", &SingleEnvManager::UseMuscle)
            .def("UseAdaptiveSampling", &SingleEnvManager::UseAdaptiveSampling)
            .def("Step", &SingleEnvManager::Step)
            .def("Reset", &SingleEnvManager::Reset)
            .def("IsEndOfEpisode", &SingleEnvManager::IsEndOfEpisode)
            .def("GetState", &SingleEnvManager::GetState)
            .def("SetAction", &SingleEnvManager::SetAction)
            .def("GetReward", &SingleEnvManager::GetReward)
            .def("Steps", &SingleEnvManager::Steps)
            .def("StepsAtOnce", &SingleEnvManager::StepsAtOnce)
            .def("GetNumTotalMuscleRelatedDofs", &SingleEnvManager::GetNumTotalMuscleRelatedDofs)
            .def("GetNumMuscles", &SingleEnvManager::GetNumMuscles)
            .def("GetMuscleTorques", &SingleEnvManager::GetMuscleTorques)
            .def("GetDesiredTorques", &SingleEnvManager::GetDesiredTorques)
            .def("SetActivationLevels", &SingleEnvManager::SetActivationLevels)
            .def("GetMuscleTuples", &SingleEnvManager::GetMuscleTuples);

#ifdef COMPILE_MULTI_ENV_MANAGER
    py::class_<EnvManager>(m, "MultiEnvManager")
            .def(py::init<const py::dict&>())
            .def("GetNumState",&EnvManager::GetNumState)
            .def("GetNumAction",&EnvManager::GetNumAction)
            .def("GetSimulationHz",&EnvManager::GetSimulationHz)
            .def("GetControlHz",&EnvManager::GetControlHz)
            .def("GetNumSteps",&EnvManager::GetNumSteps)
            .def("UseMuscle",&EnvManager::UseMuscle)
            .def("UseAdaptiveSampling",&EnvManager::UseAdaptiveSampling)
            .def("Step",&EnvManager::Step)
            .def("Reset",&EnvManager::Reset)
            .def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
            .def("GetState",&EnvManager::GetState)
            .def("SetAction",&EnvManager::SetAction)
            .def("GetReward",&EnvManager::GetReward)
            .def("Steps",&EnvManager::Steps)
            .def("StepsAtOnce",&EnvManager::StepsAtOnce)
            .def("Resets",&EnvManager::Resets)
            .def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
            .def("GetStates",&EnvManager::GetStates)
            .def("SetActions",&EnvManager::SetActions)
            .def("GetRewards",&EnvManager::GetRewards)
            .def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
            .def("GetNumMuscles",&EnvManager::GetNumMuscles)
            .def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
            .def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
            .def("SetActivationLevels",&EnvManager::SetActivationLevels)
            .def("GetMuscleTuples",&EnvManager::GetMuscleTuples);
#endif
}