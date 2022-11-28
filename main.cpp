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

MatrixXd Phi; // Final distance function
MatrixXd U; // Temperature
MatrixXd X; // Normalized gradient of U
MatrixXd B; // Integrated divergences of X
SparseMatrix<double> CotAlpha; // Matrix of cot(alpha_ij)
SparseMatrix<double> CotBeta; // Matrix of cot(beta_ij)
SparseMatrix<double> D; // Length of edges
double dt; // time step
double h; // mean spacing between adjacent nodes
MatrixXd A; // Diagonal of vertex area
SparseMatrix<double> Lc; // Laplace-Berltrami matrix, but only cotan operator

SparseMatrix<double> SparseMatrixExplicit;
SparseMatrix<double> LeftSideImplicit;
SimplicialCholesky<SparseMatrix<double>> SolverImplicit;
SimplicialCholesky<SparseMatrix<double>> SolverFinal;

int nbSteps = 1000; // number of time steps between animations
int nbStepsTotal = 1000000; // number of time steps between animations

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
		Vector3d u(V.row(i1) - V.row(i0));
		Vector3d v(V.row(i2) - V.row(i1));
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

/**
* Compute X (he)
**/
void setInitialU() {
	U = MatrixXd::Zero(V.rows(), 1);
	U(0) = 1;
}

/**
* Compute CotAlpha, CotBeta, D and dt (he)
**/
void computeAlphaBetaDdt(HalfedgeDS he) {
	std::cout << "Computing CotAlpha, CotBeta, D and dt..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	std::vector<Eigen::Triplet<double>> stackCotAlpha{};
	std::vector<Eigen::Triplet<double>> stackCotBeta{};
	std::vector<Eigen::Triplet<double>> stackD{};
	dt = 0;
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int pe = he.getOpposite(he.getNext(e)); // edge from x_k to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		int k = he.getTarget(he.getOpposite(pe)); // vertex after i
		for (int l = 0; l < vDCW; l++) {
			Vector3d ek(V.row(i) - V.row(j));
			Vector3d ej(V.row(k) - V.row(i));
			Vector3d ei(V.row(j) - V.row(k));
			stackD.push_back(Eigen::Triplet<double>(i, j, ek.norm()));
			if (dt < ek.norm())
				dt = ek.norm();
			stackCotAlpha.push_back(Eigen::Triplet<double>(i, j, 1 / tan(acos(ei.dot(-ej) / (ej.norm() + ei.norm()))))); // angle at k
			stackCotBeta.push_back(Eigen::Triplet<double>(i, k, 1 / tan(acos(ek.dot(-ei) / (ek.norm() + ei.norm()))))); // angle at j
			//stackCotBeta.push_back(Eigen::Triplet<double>(i, j, 1 / tan(acos(ej.dot(-ei) / (ej.norm() + ei.norm()))))); // angle at k //False
			//stackCotAlpha.push_back(Eigen::Triplet<double>(i, k, 1 / tan(acos(ek.dot(-ej) / (ek.norm() + ej.norm()))))); // angle at j //False
			j = he.getTarget(he.getOpposite(pe));
			pe = he.getOpposite(he.getNext(pe));
			k = he.getTarget(he.getOpposite(pe));
		}
	}
	CotAlpha = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	CotBeta = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	D = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	CotAlpha.setFromTriplets(stackCotAlpha.begin(), stackCotAlpha.end());
	CotBeta.setFromTriplets(stackCotBeta.begin(), stackCotBeta.end());
	D.setFromTriplets(stackD.begin(), stackD.end());
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for CotAlpha, CotBeta, D and dt: " << elapsed.count() << " s\n";
}

/**
* Compute h
**/
void computeh() {
	std::cout << "Computing h..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	if (D.nonZeros() == 0)
		h = 0;
	else
		h = D.sum() / D.nonZeros();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for h: " << elapsed.count() << " s\n";
}

/**
* Compute A (he)
**/
void computeA(HalfedgeDS he) {
	std::cout << "Computing A..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	A = MatrixXd::Zero(he.sizeOfVertices(), 1);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		for (int k = 0; k < vDCW; k++) {
			A(i) += D.coeffRef(i, j) * D.coeffRef(i, j) * (CotAlpha.coeffRef(i, j) + CotBeta.coeffRef(i, j));
			e = he.getOpposite(he.getNext(e));
			j = he.getTarget(he.getOpposite(e));
		}
		A(i) /= 8;
		//A(i) = 8 / A(i);
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
	std::vector<Eigen::Triplet<double>> stackL{};
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		double lii = 0;
		for (int k = 0; k < vDCW; k++) {
			double lij = (CotAlpha.coeffRef(i, j) + CotBeta.coeffRef(i, j)) / 2;
			lii -= lij;
			stackL.push_back(Eigen::Triplet<double>(i, j, lij));
			e = he.getOpposite(he.getNext(e));
			j = he.getTarget(he.getOpposite(e));
		}
		stackL.push_back(Eigen::Triplet<double>(i, i, lii));
	}
	Lc = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	Lc.setFromTriplets(stackL.begin(), stackL.end());
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for L: " << elapsed.count() << " s\n";
}

/**
* Compute SparseMatrixExplicit
**/
void computeSparseMatrixExplicit() {
	std::cout << "Computing SparseMatrixExplicit..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	SparseMatrixExplicit = dt * A.asDiagonal() * Lc;
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for SparseMatrixExplicit: " << elapsed.count() << " s\n";
}

/**
* Compute SolverImplicit
**/
void computeSolverImplicit() {
	std::cout << "Computing SolverImplicit..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	LeftSideImplicit = MatrixXd::Identity(A.rows(), A.rows()) - dt * A.asDiagonal() * Lc;
	SolverImplicit.compute(LeftSideImplicit);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for SolverImplicit: " << elapsed.count() << " s\n";
}

/**
* Compute one time step (explicit)
**/
void computeTimeStepExplicit() {
	//std::cout << "Computing one time step (explicit)..." << std::endl;
	//auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	U += SparseMatrixExplicit * U;
	//auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Computing time for one time step (explicit): " << elapsed.count() << " s\n";
}

/**
* Compute one time step (implicit)
**/
void computeTimeStepImplicit() {
	//std::cout << "Computing one time step (implicit)..." << std::endl;
	//auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	U = SolverImplicit.solve(U);
	//auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Computing time for one time step (implicit): " << elapsed.count() << " s\n";
}

/**
* Compute X
**/
void computeX(HalfedgeDS he) {
	std::cout << "Computing X..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	MatrixXd Seen = MatrixXd::Zero(F.rows(), 1);
	X = MatrixXd::Zero(F.rows(), 3);
	for (int e = 0; e < he.sizeOfHalfedges(); e++) {
		int f = he.getFace(e);
		if (f >= 0)			
			if (Seen(f) == 0) {
				Seen(f) = 1;
				int i = he.getTarget(e);
				int j = he.getTarget(he.getNext(e));
				int k = he.getTarget(he.getPrev(e));
				Vector3d ei(V.row(k) - V.row(j)); // vector opposite to the vertex at the end of i
				Vector3d ej(V.row(i) - V.row(k)); // vector opposite to the vertex at the end of j
				Vector3d ek(V.row(j) - V.row(i)); // vector opposite to the vertex at the end of k
				Vector3d N = ei.cross(-ek).normalized();
				X.row(f) += U(i) * N.cross(ei);
				X.row(f) += U(j) * N.cross(ej);
				X.row(f) += U(k) * N.cross(ek);
				if (X.row(f).norm() > 0)
					X.row(f).normalize();
			}	
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for X: " << elapsed.count() << " s\n";
}

/**
* Compute B
**/
void computeB(HalfedgeDS he) {
	std::cout << "Computing B..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	B = MatrixXd::Zero(he.sizeOfVertices(), 1);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_2 to x_i
		int pe = he.getOpposite(he.getNext(e)); // edge from x_1 to x_i
		int v1 = he.getTarget(he.getOpposite(e)); // vertex after i
		int v2 = he.getTarget(he.getOpposite(pe)); // vertex before i
		int f = he.getFace(e);
		for (int k = 0; k < vDCW; k++) {
			if (f >= 0) {
				Vector3d e1(V.row(v2) - V.row(i));
				Vector3d e2(V.row(v1) - V.row(i));
				Vector3d x12(X.row(f));
				B(i) += (CotBeta.coeffRef(i, v2) * e1.dot(x12) + CotAlpha.coeffRef(i, v1) * e2.dot(x12)) / 2;
			}
			v1 = he.getTarget(he.getOpposite(pe));
			pe = he.getOpposite(he.getNext(pe));
			v2 = he.getTarget(he.getOpposite(pe));
			f = he.getFace(pe);
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for B: " << elapsed.count() << " s\n";
}

/**
* Compute Phi
**/
void computePhi() {
	std::cout << "Computing Phi..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	SolverFinal.compute(Lc);
	Phi = SolverFinal.solve(B);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for Phi: " << elapsed.count() << " s\n";
}


// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	switch (key) {
	case '1':
	{
		MatrixXd C;
		igl::jet(Phi, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '2':
	{
		setInitialU();
		MatrixXd C;
		igl::jet(U, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '3':
		viewer.data().set_normals(N_vertices);
		return true;
	case '4':
		viewer.data().set_normals(he_N_vertices);
		return true;
	case '5':
	{
		computeTimeStepExplicit();
		MatrixXd C;
		igl::jet(U, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '6':
	{
		for (int i = 0; i < nbSteps; i++)
			computeTimeStepExplicit();
		MatrixXd C;
		igl::jet(U, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '7':
	{
		igl::opengl::glfw::Viewer viewer;
		viewer.data().show_lines = false;
		viewer.data().set_mesh(V, F);
		viewer.data().set_normals(N_faces);
		viewer.core().is_animating = true;
		int j = 0;
		viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&)->bool // run animation
		{
			for (int i = 0; i < nbSteps; i++)
				computeTimeStepExplicit();
			j++;
			std::cout << j << std::endl;
			MatrixXd C;
			igl::jet(U, true, C); // Assign per-vertex colors
			viewer.data().set_colors(C); // Add per-vertex colors
			return false; };
		viewer.launch(); // run the editor
		return true;
	}
	case '8':
	{
		computeTimeStepImplicit();
		MatrixXd C;
		igl::jet(U, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '9':
	{
		for (int i = 0; i < nbSteps; i++)
			computeTimeStepImplicit();
		MatrixXd C;
		igl::jet(U, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '0':
	{
		igl::opengl::glfw::Viewer viewer;
		viewer.data().show_lines = false;
		viewer.data().set_mesh(V, F);
		viewer.data().set_normals(N_faces);
		viewer.core().is_animating = true;
		viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&)->bool // run animation
		{
			for (int i = 0; i < nbSteps; i++)
				computeTimeStepImplicit();
			MatrixXd C;
			igl::jet(U, true, C); // Assign per-vertex colors
			viewer.data().set_colors(C); // Add per-vertex colors
			return false; };
		viewer.launch(); // run the editor
	}
	default: break;
	}
	return false;
}

// ------------ main program ----------------
int main(int argc, char *argv[]) {

//	if(argc<2) {
//		std::cout << "Error: input file required (.OFF)" << std::endl;
//		return 0;
//	}
//	std::cout << "reading input file: " << argv[1] << std::endl;

	//igl::readOFF(argv[1], V, F);
 	//igl::readOFF("../data/cube_open.off", V, F);	// 1 boundary
	//igl::readOFF("../data/cube_tri.off", V, F);	// 0 boundary
	//igl::readOFF("../data/star.off", V, F);		// 0 boundary
	igl::readOFF("../data/sphere.off", V, F);		// 0 boundary
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

	HalfedgeBuilder* builder=new HalfedgeBuilder();

	HalfedgeDS he=builder->createMesh(V.rows(), F);

	// New for the project

	rescale();
	setInitialU();
	computeAlphaBetaDdt(he);
	std::cout << "dt: " << dt << std::endl;
	computeh();
	std::cout << "h: " << h << std::endl;
	computeA(he);
	computeL(he);
	computeSparseMatrixExplicit();
	computeSolverImplicit();

	// correct number of steps:
	// - 1000000 for sphere
	// - 2000000 for nefertiti ?
	// - 800 for star ?
	for (int i = 0; i < nbStepsTotal; i++)
		computeTimeStepExplicit();
	computeX(he);
	computeB(he);
	computePhi();
	setInitialU();

	// Compute normals

	// Compute per-vertex normals
	igl::per_vertex_normals(V,F,N_vertices);

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
		"Press '1' to compute the distance" << std::endl <<
		"Press '2' to reset the temperature" << std::endl <<
		"Press '3' for per-vertex normals calling pre-defined functions of LibiGL" << std::endl <<
		"Press '4' for HE_per-vertex normals using HalfEdge structure" << std::endl <<
		"Press '5' to compute one steps (explicit)" << std::endl <<
		"Press '6' to compute " << nbSteps << " steps (explicit)" << std::endl <<
		"Press '7' to animate (explicit)" << std::endl <<
		"Press '8' to compute one steps (implicit)" << std::endl <<
		"Press '9' to compute " << nbSteps << " steps (implicit)" << std::endl <<
		"Press '0' to animate (implicit)" << std::endl;

	MatrixXd C;
	igl::jet(U,true,C); // Assign per-vertex colors
	viewer.data().set_colors(C); // Add per-vertex colors

	//viewer.core(0).align_camera_center(V, F); //not needed
	viewer.launch(); // run the viewer
}