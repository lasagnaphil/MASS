#include "EnvManager.h"
#include "DARTHelper.h"
#include <thread>

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs), mEnvs(num_envs)
{
	dart::math::seedRand();

    std::vector<std::thread> threads;

	for (int i = 0; i < mNumEnvs; i++) {
	    mEnvs[i] = new MASS::Environment();
	    MASS::Environment* env = mEnvs[i];
	    threads.emplace_back([meta_file, env]() {
	        env->Initialize(meta_file, false);

            // env->SetUseMuscle(false);
            // env->SetControlHz(30);
            // env->SetSimulationHz(600);
            // env->SetRewardParameters(0.65,0.1,0.15,0.1);

            // MASS::Character* character = new MASS::Character();
            // character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"),false);
            // if(env->GetUseMuscle())
            // 	character->LoadMuscles(std::string(MASS_ROOT_DIR)+std::string("/data/muscle.xml"));

            // character->LoadBVH(std::string(MASS_ROOT_DIR)+std::string("/data/motion/walk.bvh"),true);

            // double kp = 300.0;
            // character->SetPDParameters(kp,sqrt(2*kp));
            // env->SetCharacter(character);
            // env->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

            // env->Initialize();
	    });
    }
	for (int i = 0; i < mNumEnvs; i++) {
	    threads[i].join();
	}
}
int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}
int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}
int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}
int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}
int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}
bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}
void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}
void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}
bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}
py::array_t<float>
EnvManager::
GetState(int id)
{
	return toNumPyArray(mEnvs[id]->GetState());
}
void 
EnvManager::
SetAction(py::array_t<float> np_array, int id)
{
	mEnvs[id]->SetAction(toEigenVector(np_array));
}
double 
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

void
EnvManager::
Steps(int num)
{
    std::vector<std::thread> threads;

	for (int id = 0; id < mNumEnvs; id++)
	{
        MASS::Environment* env = mEnvs[id];
	    threads.emplace_back([num, env](){
	        for (int j = 0; j < num; j++) {
                env->Step();
            }
	    });
	}

    for (int id = 0; id < mNumEnvs; id++) {
        threads[id].join();
    }
}
void
EnvManager::
StepsAtOnce() {
    std::vector<std::thread> threads;

    int num = this->GetNumSteps();
    for (int id = 0; id < mNumEnvs; id++) {
        MASS::Environment* env = mEnvs[id];
        threads.emplace_back([num, env](){
            for (int j = 0; j < num; j++) {
                env->Step();
            }
        });
    }

    for (int id = 0; id < mNumEnvs; id++) {
        threads[id].join();
    }
}
void
EnvManager::
Resets(bool RSI)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}
py::array_t<float>
EnvManager::
IsEndOfEpisodes()
{
	std::vector<bool> is_end_vector(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		is_end_vector[id] = mEnvs[id]->IsEndOfEpisode();
	}

	return toNumPyArray(is_end_vector);
}
py::array_t<float>
EnvManager::
GetStates()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState());
	for (int id = 0;id<mNumEnvs;++id)
	{
		states.row(id) = mEnvs[id]->GetState().transpose();
	}

	return toNumPyArray(states);
}
void
EnvManager::
SetActions(py::array_t<float> np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(action.row(id).transpose());
	}
}
py::array_t<float>
EnvManager::
GetRewards()
{
	std::vector<float> rewards(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		rewards[id] = mEnvs[id]->GetReward();
	}
	return toNumPyArray(rewards);
}
py::array_t<float>
EnvManager::
GetMuscleTorques()
{
    std::vector<std::thread> threads;
    std::vector<Eigen::VectorXd> mt(mNumEnvs);

	for (int id = 0; id < mNumEnvs; ++id)
	{
	    MASS::Environment* env = mEnvs[id];
	    Eigen::VectorXd* mt_ptr = &mt[id];
	    threads.emplace_back([env, mt_ptr](){
            *mt_ptr = env->GetMuscleTorques();
	    });
	}

    for (int id = 0; id < mNumEnvs; id++) {
        threads[id].join();
    }

	return toNumPyArray(mt);
}
py::array_t<float>
EnvManager::
GetDesiredTorques()
{
    std::vector<std::thread> threads;
	std::vector<Eigen::VectorXd> tau_des(mNumEnvs);
	
	for (int id = 0; id < mNumEnvs; ++id)
	{
	    MASS::Environment* env = mEnvs[id];
	    Eigen::VectorXd* tau_des_ptr = &tau_des[id];
	    threads.emplace_back([env, tau_des_ptr](){
            *tau_des_ptr = env->GetDesiredTorques();
	    });
	}

    for (int id = 0; id < mNumEnvs; id++) {
        threads[id].join();
    }

	return toNumPyArray(tau_des);
}

void
EnvManager::
SetActivationLevels(py::array_t<float> np_array)
{
	std::vector<Eigen::VectorXd> activations =toEigenVectorVector(np_array);
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations[id]);
}

py::list
EnvManager::
GetMuscleTuples()
{
	py::list all;
	for (int id = 0; id < mNumEnvs; ++id)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			py::list t;
			t.append(toNumPyArray(tps[j].JtA));
			t.append(toNumPyArray(tps[j].tau_des));
			t.append(toNumPyArray(tps[j].L));
			t.append(toNumPyArray(tps[j].b));
			all.append(t);
		}
		tps.clear();
	}

	return all;
}

PYBIND11_MODULE(pymss, m)
{
	py::class_<EnvManager>(m, "EnvManager")
        .def(py::init<std::string,int>())
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("UseMuscle",&EnvManager::UseMuscle)
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
}