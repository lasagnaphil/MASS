#include "NumPyHelper.h"

py::array_t<float> toNumPyArray(const std::vector<float>& val)
{
    size_t n = val.size();
    py::array_t<float> array(n);
    py::buffer_info buf = array.request();

    std::memcpy(buf.ptr, val.data(), n * sizeof(float));
	return array;
}
py::array_t<float> toNumPyArray(const std::vector<double>& val)
{
    size_t n = val.size();
    py::array_t<float> array(n);
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

    for (int i = 0; i < n; i++) {
        ptr[i] = (float)val[i];
    }
	return array;
}
py::array_t<float> toNumPyArray(const std::vector<Eigen::VectorXd>& val)
{
	int n = val.size();
	int m = val[0].rows();
	py::array_t<float> array({n, m});
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
	    for (int j = 0; j < m; j++) {
	        ptr[i*m+j] = (float)val[i](j);
	    }
	}

	return array;	
}
py::array_t<float> toNumPyArray(const std::vector<Eigen::MatrixXd>& val)
{
	int n = val.size();
	int m = val[0].rows();
	int l = val[0].cols();

    py::array_t<float> array({n, m, l});
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++) {
        for (int j = 0; j < m; j++)
            for (int k = 0; k < l; k++)
                ptr[i*m*l + j*l + k] = (float)val[i](j,k);
    }

	return array;
}
py::array_t<float> toNumPyArray(const std::vector<std::vector<float>>& val)
{
	int n = val.size();
	int m = val[0].size();

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++)
	{
        std::memcpy(ptr + i*m, val[i].data(), m*sizeof(float));
	}

	return array;
}
py::array_t<float> toNumPyArray(const std::vector<std::vector<double>>& val)
{
	int n = val.size();
	int m = val[0].size();

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
		    ptr[i*m+j] = (float)val[i][j];
		}
	}

	return array;
}
//always return 1-dim array
py::array_t<float> toNumPyArray(const std::vector<bool>& val)
{
	int n = val.size();

    py::array_t<float> array(n);
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
        ptr[i] = (float)val[i];
	}

	return array;
}

//always return 1-dim array
py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();

    py::array_t<float> array(n);
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	for(int i =0;i<n;i++)
	{
        ptr[i] = (float)vec(i);
	}

	return array;
}
//always return 2-dim array
py::array_t<float> toNumPyArray(const Eigen::MatrixXd& matrix)
{
	int n = matrix.rows();
	int m = matrix.cols();

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request();
    float* ptr = reinterpret_cast<float*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
            ptr[i*m+j] = (float)matrix(i,j);
		}
	}

	return array;
}
//always return 2-dim array
py::array_t<float> toNumPyArray(const Eigen::Isometry3d& T)
{
	int n = 4;
	int m = 4;

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request();
	float* dest = reinterpret_cast<float*>(buf.ptr);

	Eigen::Matrix3d R = T.linear();
	Eigen::Vector3d p = T.translation();
	dest[0] = T(0,0),dest[1] = T(0,1),dest[2] = T(0,2),dest[3] = p[0];
	dest[4] = T(1,0),dest[5] = T(1,1),dest[6] = T(1,2),dest[7] = p[1];
	dest[8] = T(2,0),dest[9] = T(2,1),dest[10] = T(2,2),dest[11] = p[2];
	dest[12] = 0.0,dest[13] = 0.0,dest[14] = 0.0,dest[15] = 1.0;

	return array;
}
Eigen::VectorXd toEigenVector(const py::array_t<float>& array)
{
	Eigen::VectorXd vec(array.shape(0));

    py::buffer_info buf = array.request();
	float* srcs = reinterpret_cast<float*>(buf.ptr);

	for(int i=0;i<array.shape(0);i++)
	{
		vec(i) = (double)srcs[i];
	}
	return vec;
}
std::vector<Eigen::VectorXd> toEigenVectorVector(const py::array_t<float>& array)
{
	std::vector<Eigen::VectorXd> mat;
	mat.resize(array.shape(0));

    py::buffer_info buf = array.request();
	float* srcs = reinterpret_cast<float*>(buf.ptr);
	int n = array.shape(0);
	int m = array.shape(1);

	for(int i=0;i<n;i++){
		mat[i].resize(m);
		for(int j=0;j<m;j++)
			mat[i](j) = (double)srcs[i*m+j];
	}

	return mat;	
}
Eigen::MatrixXd toEigenMatrix(const py::array_t<float>& array)
{
	Eigen::MatrixXd mat(array.shape(0),array.shape(1));

    py::buffer_info buf = array.request();
	float* srcs = reinterpret_cast<float*>(buf.ptr);
    int n = array.shape(0);
    int m = array.shape(1);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat(i,j) = srcs[i*m+j];
		}
	}
	return mat;
}
std::vector<bool> toStdVector(const py::list& list)
{
	std::vector<bool> vec(py::len(list));
	for(int i =0;i<vec.size();i++)
	    vec[i] = list[i].cast<bool>();
	return vec;
}