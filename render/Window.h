#ifndef __MASS_WINDOW_H__
#define __MASS_WINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"

#include <pybind11/embed.h>
#include <pybind11/pytypes.h>

namespace MASS
{
class Environment;
class Muscle;
class Window : public dart::gui::glut::Win3D
{
public:
	Window(Environment* env);
	Window(Environment* env,const std::string& nn_path);
	Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path);
	~Window();

	void draw() override;
	void keyboard(unsigned char _key, int _x, int _y) override;
	void displayTimer(int _val) override;
private:
	void SetFocusing();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);

	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawGround(double y);
	void Step();
	void Reset();

	Eigen::VectorXd GetActionFromNN();
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

	pybind11::object load_module;
	pybind11::object mm,mns,sys_module,nn_module,muscle_nn_module;

    Environment* mEnv;
	bool mFocus;
	bool mSimulating;
	bool mDrawOBJ;
	bool mDrawShadow;
	bool mNNLoaded;
	bool mMuscleNNLoaded;
	Eigen::Affine3d mViewMatrix;
};
};


#endif