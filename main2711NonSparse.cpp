#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <igl/readOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>

#include <igl/gaussian_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>

#include "HalfedgeBuilder.cpp"

using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

MatrixXd V;
MatrixXi F;

MatrixXd X; // vector of vertices
MatrixXd CotAlpha; // Matrix of cot(alpha_ij)
MatrixXd CotBeta; // Matrix of cot(beta_ij)
MatrixXd D; // Length of edges
double dt; // time step
MatrixXd Ainv; // Matrix of vertex area
MatrixXd AinvVec; // Diagonal of vertex area
MatrixXd L; // Laplace-Berltrami matrix

int nbSteps = 1000; // number of time steps between animations

MatrixXd N_faces; // computed calling pre-defined functions of LibiGL
MatrixXd N_vertices; // computed calling pre-defined functions of LibiGL
MatrixXd lib_N_vertices; // computed using face-vertex structure of LibiGL
MatrixXi lib_Deg_vertices; // computed using face-vertex structure of LibiGL
MatrixXd he_N_vertices; // computed using the HalfEdge data structure

/**
* Rescale the mesh in [0,1]^3
**/
void rescale() {
	std::cout << "Rescaling..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	double minx = V.row(0)[0];
	double maxx = V.row(0)[0];
	double miny = V.row(0)[1];
	double maxy = V.row(0)[1];
	double minz = V.row(0)[2];
	double maxz = V.row(0)[2];
	for (int i = 0; i < V.rows(); i++) {
		minx = min(minx, V.row(i)[0]);
		maxx = max(maxx, V.row(i)[0]);
		miny = min(miny, V.row(i)[1]);
		maxy = max(maxy, V.row(i)[1]);
		minz = min(minz, V.row(i)[2]);
		maxz = max(maxz, V.row(i)[2]);
	}
	double lenx = maxx - minx;
	double leny = maxy - miny;
	double lenz = maxz - minz;
	for (int i = 0; i < V.rows(); i++) {
		V.row(i)[0] = (V.row(i)[0] + minx) / lenx;
		V.row(i)[1] = (V.row(i)[1] + miny) / leny;
		V.row(i)[2] = (V.row(i)[2] + minz) / lenz;
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for rescaling: " << elapsed.count() << " s\n";
}

/**
* Return the degree of a given vertex 'v'
* Turn around vertex 'v' in CCW order
**/
int vertexDegreeCCW(HalfedgeDS he, int v) {
	int vDCCW = 1;
	int e = he.getEdge(v);
	int pe = he.getOpposite(he.getNext(e));
	while (pe != e) {
		vDCCW++;
		pe = he.getOpposite(he.getNext(pe));
	}
	return vDCCW;
}

/**
* Compute the vertex normals (he)
**/
void vertexNormals(HalfedgeDS he) {
	std::cout << "Computing the vertex normals using vertexNormals..." << std::endl;
	auto starthe = std::chrono::high_resolution_clock::now(); // for measuring time performances
	he_N_vertices = MatrixXd::Zero(he.sizeOfVertices(), 3);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int i0 = he.getTarget(he.getOpposite(he.getEdge(i)));
		int i1 = i;
		int i2 = he.getTarget(he.getNext(he.getEdge(i)));
		Vector3d u(V.row(i1)[0] - V.row(i0)[0], V.row(i1)[1] - V.row(i0)[1], V.row(i1)[2] - V.row(i0)[2]);
		Vector3d v(V.row(i2)[0] - V.row(i1)[0], V.row(i2)[1] - V.row(i1)[1], V.row(i2)[2] - V.row(i1)[2]);
		Vector3d w = u.cross(v);
		w.normalize();
		MatrixXd n = MatrixXd::Zero(1, 3);
		n(0) = w[0]; n(1) = w[1]; n(2) = w[2];
		he_N_vertices.row(i0) += n;
		he_N_vertices.row(i1) += n;
		he_N_vertices.row(i2) += n;
	}
	for (int i = 0; i < V.rows(); i++)
		he_N_vertices.row(i).normalize();
	auto finishhe = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedhe = finishhe - starthe;
	std::cout << "Computing time using vertexNormals: " << elapsedhe.count() << " s\n";
}

// Compute lib_per-vertex normals
/**
* Compute the vertex normals (global, using libiGl data structure)
**/
void lib_vertexNormals() {
	std::cout << "Computing the vertex normals using lib_vertexNormals..." << std::endl;
	auto startlib = std::chrono::high_resolution_clock::now(); // for measuring time performances
	lib_N_vertices = MatrixXd::Zero(V.rows(), 3);
	for (int i = 0; i < F.rows(); i++) {
		int i0 = F.row(i)[0];
		int i1 = F.row(i)[1];
		int i2 = F.row(i)[2];
		Vector3d u(V.row(i1)[0] - V.row(i0)[0], V.row(i1)[1] - V.row(i0)[1], V.row(i1)[2] - V.row(i0)[2]);
		Vector3d v(V.row(i2)[0] - V.row(i1)[0], V.row(i2)[1] - V.row(i1)[1], V.row(i2)[2] - V.row(i1)[2]);
		Vector3d w = u.cross(v);
		w.normalize();
		MatrixXd n = MatrixXd::Zero(1, 3);
		n(0) = w[0]; n(1) = w[1]; n(2) = w[2];
		lib_N_vertices.row(i0) += n;
		lib_N_vertices.row(i1) += n;
		lib_N_vertices.row(i2) += n;
	}
	for (int i = 0; i < V.rows(); i++)
		lib_N_vertices.row(i).normalize();
	auto finishlib = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedlib = finishlib - startlib;
	std::cout << "Computing time using lib_vertexNormals: " << elapsedlib.count() << " s\n";
}


/**
* Compute X (he)
**/
void computeX(HalfedgeDS he) {
	X = MatrixXd::Zero(he.sizeOfVertices(), 1);
	X(0) = 1;
}

/**
* Compute CotAlpha, CotBeta, D and dt (he)
**/
void computeAlphaBetaDdt(HalfedgeDS he) {
	std::cout << "Computing CotAlpha, CotBeta, D and dt..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	CotAlpha = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	CotBeta = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	D = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	dt = 0;
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int pe = he.getOpposite(he.getNext(e)); // edge from x_k to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		int k = he.getTarget(he.getOpposite(pe)); // vertex after i
		for (int l = 0; l < vDCW; l++) {
			Vector3d u(V.row(i)[0] - V.row(j)[0], V.row(i)[1] - V.row(j)[1], V.row(i)[2] - V.row(j)[2]);
			Vector3d v(V.row(k)[0] - V.row(i)[0], V.row(k)[1] - V.row(i)[1], V.row(k)[2] - V.row(i)[2]);
			Vector3d w(V.row(j)[0] - V.row(k)[0], V.row(j)[1] - V.row(k)[1], V.row(j)[2] - V.row(k)[2]);
			D(i, j) = u.norm();
			if (dt < D(i, j))
				dt = D(i, j);
			CotAlpha(i, j) = 1 / tan(acos(w.dot(-v) / (v.norm() + w.norm()))); // angle at k
			CotBeta(i, k) = 1 / tan(acos(u.dot(-w) / (u.norm() + w.norm()))); // angle at j
			j = he.getTarget(he.getOpposite(pe));
			pe = he.getOpposite(he.getNext(pe));
			k = he.getTarget(he.getOpposite(pe));
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for CotAlpha, CotBeta, D and dt: " << elapsed.count() << " s\n";
}

/**
* Compute A (he)
**/
void computeA(HalfedgeDS he) {
	std::cout << "Computing A..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	Ainv = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	AinvVec = MatrixXd::Zero(he.sizeOfVertices(), 1);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		for (int k = 0; k < vDCW; k++) {
			Ainv(i, i) += D(i, j) * D(i, j) * (CotAlpha(i, j) + CotBeta(i, j));
			e = he.getOpposite(he.getNext(e));
			j = he.getTarget(he.getOpposite(e));
		}
		Ainv(i, i) = 8 / Ainv(i, i);
		AinvVec(i) = 8 / Ainv(i, i);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for A: " << elapsed.count() << " s\n";
}

/**
* Compute L (he)
**/
void computeL(HalfedgeDS he) {
	std::cout << "Computing L..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	L = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		for (int k = 0; k < vDCW; k++) {
			L(i, j) = (CotAlpha(i, j) + CotBeta(i, j)) / 2;
			L(i, i) -= L(i, j);
			e = he.getOpposite(he.getNext(e));
			j = he.getTarget(he.getOpposite(e));
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for L: " << elapsed.count() << " s\n";
}

/**
* Compute one time dtep
**/
void computeTimeStepExplicit() {
	//std::cout << "Computing one time step..." << std::endl;
	//auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	X = X + dt * AinvVec.asDiagonal() * L * X;
	//for (int i = 0; i < 8; i++)
	//	std::cout << X(i) << std::endl;
	//auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Computing time for one time step: " << elapsed.count() << " s\n";
}

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	switch (key) {
	case '1':
		viewer.data().set_normals(N_faces);
		return true;
	case '2':
		viewer.data().set_normals(N_vertices);
		return true;
	case '3':
		viewer.data().set_normals(lib_N_vertices);
		return true;
	case '4':
		viewer.data().set_normals(he_N_vertices);
		return true;
	case '5':
	{
		computeTimeStepExplicit();
		MatrixXd C;
		igl::jet(X, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
	}
	case '6':
	{
		for (int i = 0; i < nbSteps; i++)
			computeTimeStepExplicit();
		MatrixXd C;
		igl::jet(X, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
	}
	case '7':
	{
		igl::opengl::glfw::Viewer viewer;
		viewer.data().show_lines = false;
		viewer.data().set_mesh(V, F);
		viewer.data().set_normals(N_faces);
		viewer.core().is_animating = true;
		viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&)->bool // run animation
		{
			for (int i = 0; i < nbSteps; i++)
				computeTimeStepExplicit();
			MatrixXd C;
			igl::jet(X, true, C); // Assign per-vertex colors
			viewer.data().set_colors(C); // Add per-vertex colors
			return false; };
		viewer.launch(); // run the editor

	}
	default: break;
	}
	return false;
}

// ------------ main program ----------------
int main(int argc, char* argv[]) {

	//	if(argc<2) {
	//		std::cout << "Error: input file required (.OFF)" << std::endl;
	//		return 0;
	//	}
	//	std::cout << "reading input file: " << argv[1] << std::endl;

	//igl::readOFF(argv[1], V, F);
	//igl::readOFF("../data/cube_open.off", V, F);	// 1 boundary
	//igl::readOFF("../data/cube_tri.off", V, F);	// 0 boundary
	//igl::readOFF("../data/star.off", V, F);		// 0 boundary
	igl::readOFF("../data/sphere.off", V, F);
	//igl::readOFF("../data/nefertiti.off", V, F);	// 1 boundary
	//igl::readOFF("../data/cat0.off",V,F);			// 2 boundaries
	//igl::readOFF("../data/chandelier.off", V, F);	// 10 boundaries
	//igl::readOFF("../data/face.off", V, F);		// 1 boundary
	//igl::readOFF("../data/high_genus.off", V, F);	// 0 boundary
	//igl::readOFF("../data/homer.off", V, F);		// 1 boundary
	//igl::readOFF("../data/venus.off", V, F);		// 1 boundary
	//igl::readOFF("../data/bunny.off", V, F);
	//igl::readOFF("../data/egea.off", V, F);
	//igl::readOFF("../data/gargoyle_tri.off", V, F);

	//print the number of mesh elements
	std::cout << "Points: " << V.rows() << std::endl;

	HalfedgeBuilder* builder = new HalfedgeBuilder();

	HalfedgeDS he = builder->createMesh(V.rows(), F);

	// New for the project

	rescale();
	computeX(he);
	computeAlphaBetaDdt(he);
	std::cout << "dt: " << dt << std::endl;
	computeA(he);
	computeL(he);

	// Compute normals

	// Compute per-face normals
	igl::per_face_normals(V, F, N_faces);

	// Compute per-vertex normals
	igl::per_vertex_normals(V, F, N_vertices);

	// Compute lib_per-vertex normals
	lib_vertexNormals();

	// Compute he_per-vertex normals
	vertexNormals(he);

	///////////////////////////////////////////

		// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V, F);
	viewer.data().set_normals(N_faces);
	std::cout <<
		"Press '1' for per-face normals calling pre-defined functions of LibiGL." << std::endl <<
		"Press '2' for per-vertex normals calling pre-defined functions of LibiGL." << std::endl <<
		"Press '3' for lib_per-vertex normals using face-vertex structure of LibiGL ." << std::endl <<
		"Press '4' for HE_per-vertex normals using HalfEdge structure." << std::endl <<
		"Press '5' to compute one steps" << std::endl <<
		"Press '6' to compute " << nbSteps << " steps" << std::endl <<
		"Press '7' to animate" << std::endl;

	MatrixXd C;
	igl::jet(X, true, C); // Assign per-vertex colors
	viewer.data().set_colors(C); // Add per-vertex colors

	//viewer.core(0).align_camera_center(V, F); //not needed
	viewer.launch(); // run the viewer
}