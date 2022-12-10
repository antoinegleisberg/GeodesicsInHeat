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

MatrixXd GeodesicDistance; // Final distance function
MatrixXd Temperature; // Temperature, called U in the paper
MatrixXd NormalizedTemperatureGradient; // Normalized gradient of Temperature, called X in the paper
MatrixXd B; // Integrated divergences of NormalizedTemperatureGradient
SparseMatrix<double> CotAlpha; // Matrix of cot(alpha_ij)
SparseMatrix<double> CotBeta; // Matrix of cot(beta_ij)
SparseMatrix<double> Distances; // Length of edges
double dt; // time step
double h; // mean spacing between adjacent nodes
MatrixXd A; // Distancesiagonal of vertex area; Voronoi area of each vertex
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
	VectorXd minV = V.colwise().minCoeff();
	VectorXd maxV = V.colwise().maxCoeff();
	VectorXd range = maxV - minV;
	V = (V.rowwise() - minV.transpose());
	for (int i = 0; i < 3; i++) {
		V.col(i) = V.col(i) / range(i);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for rescaling: " << elapsed.count() << " s\n";
	return;
}

/**
* Return the degree of a given vertex 'v'
* Turn around vertex 'v' in CCW order
**/
int vertexDegreeCCW(HalfedgeDS he, int v) {
	int vertexDegree = 1;
	int e = he.getEdge(v);
	int nextEdge = he.getOpposite(he.getNext(e));
	while (nextEdge != e) {
		vertexDegree++;
		nextEdge = he.getOpposite(he.getNext(nextEdge));
	}
	return vertexDegree;
}

/**
* Compute the vertex normals (he)
**/
void vertexNormals(HalfedgeDS he) {
	std::cout << "Computing the vertex normals using vertexNormals..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	he_N_vertices = MatrixXd::Zero(he.sizeOfVertices(), 3);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int i0 = he.getTarget(he.getOpposite(he.getEdge(i)));
		int i1 = i;
		int i2 = he.getTarget(he.getNext(he.getEdge(i)));
		Vector3d u(V.row(i1) - V.row(i0));
		Vector3d v(V.row(i2) - V.row(i1));
		Vector3d w = u.cross(v);
		w.normalize();
		he_N_vertices.row(i0) += w;
		he_N_vertices.row(i1) += w;
		he_N_vertices.row(i2) += w;
		
		// @Aude : ok pour ce changement ?
		/*
		w.normalize();
		MatrixXd n = MatrixXd::Zero(1, 3);
		n(0) = w[0]; n(1) = w[1]; n(2) = w[2];
		he_N_vertices.row(i0) += n;
		he_N_vertices.row(i1) += n;
		he_N_vertices.row(i2) += n;
		*/
	}
	he_N_vertices.rowwise().normalize();
	// @Aude : ok pour ce changement ?
	// for (int i = 0; i < V.rows(); i++) he_N_vertices.row(i).normalize();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time using vertexNormals: " << elapsed.count() << " s" << std::endl;
}

/**
* Initialise Temperature matrix
**/
void setInitialTemperature() {
	Temperature = MatrixXd::Zero(V.rows(), 1);
	Temperature(0) = 1;
}

/**
* Compute CotAlpha, CotBeta, Distances and dt (he)
**/
void computeAlphaBetaDdt(HalfedgeDS he) {
	std::cout << "Computing CotAlpha, CotBeta, Distances and dt..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<Eigen::Triplet<double>> stackCotAlpha{};
	std::vector<Eigen::Triplet<double>> stackCotBeta{};
	std::vector<Eigen::Triplet<double>> stackD{};
	dt = 0;
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int e = he.getEdge(i); // edge pointing from j towards i
		int nextEdge = he.getOpposite(he.getNext(e)); // edge pointing from k towards i
		int j = he.getTarget(he.getOpposite(e));
		int k = he.getTarget(he.getOpposite(nextEdge));
		for (int l = 0; l < vertexDegreeCCW(he, i); l++) {
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
			
			j = k;
			nextEdge = he.getOpposite(he.getNext(nextEdge));
			k = he.getTarget(he.getOpposite(nextEdge));
		}
	}
	CotAlpha = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	CotBeta = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	Distances = SparseMatrix<double>(he.sizeOfVertices(), he.sizeOfVertices());
	CotAlpha.setFromTriplets(stackCotAlpha.begin(), stackCotAlpha.end());
	CotBeta.setFromTriplets(stackCotBeta.begin(), stackCotBeta.end());
	Distances.setFromTriplets(stackD.begin(), stackD.end());
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for CotAlpha, CotBeta, Distances and dt: " << elapsed.count() << " s\n";
}

/**
* Compute h as the mean of the edge lengths
**/
void computeh() {
	std::cout << "Computing h..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	if (Distances.nonZeros() == 0)
		h = 0;
	else
		h = Distances.sum() / Distances.nonZeros();
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
		int e = he.getEdge(i); // edge from j to i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		for (int k = 0; k < vDCW; k++) {
			A(i) += Distances.coeffRef(i, j) * Distances.coeffRef(i, j) * (CotAlpha.coeffRef(i, j) + CotBeta.coeffRef(i, j));
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
	Temperature += SparseMatrixExplicit * Temperature;
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
	// @ Aude : pourquoi pas += pour celui la ?
	Temperature = SolverImplicit.solve(Temperature);
	//auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Computing time for one time step (implicit): " << elapsed.count() << " s\n";
}

/**
* Compute NormalizedTemperatureGradient
**/
void computeNormalizedTemperatureGradient(HalfedgeDS he) {
	std::cout << "Computing NormalizedTemperatureGradient..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	MatrixXd Seen = MatrixXd::Zero(F.rows(), 1);
	NormalizedTemperatureGradient = MatrixXd::Zero(F.rows(), 3);
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
				NormalizedTemperatureGradient.row(f) += Temperature(i) * N.cross(ei);
				NormalizedTemperatureGradient.row(f) += Temperature(j) * N.cross(ej);
				NormalizedTemperatureGradient.row(f) += Temperature(k) * N.cross(ek);
				if (NormalizedTemperatureGradient.row(f).norm() > 0)
					NormalizedTemperatureGradient.row(f).normalize();
			}	
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for NormalizedTemperatureGradient: " << elapsed.count() << " s\n";
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
		int nextEdge = he.getOpposite(he.getNext(e)); // edge from x_1 to x_i
		int v1 = he.getTarget(he.getOpposite(e)); // vertex after i
		int v2 = he.getTarget(he.getOpposite(nextEdge)); // vertex before i
		int f = he.getFace(e);
		for (int k = 0; k < vDCW; k++) {
			if (f >= 0) {
				Vector3d e1(V.row(v2) - V.row(i));
				Vector3d e2(V.row(v1) - V.row(i));
				Vector3d x12(NormalizedTemperatureGradient.row(f));
				B(i) += (CotBeta.coeffRef(i, v2) * e1.dot(x12) + CotAlpha.coeffRef(i, v1) * e2.dot(x12)) / 2;
			}
			v1 = he.getTarget(he.getOpposite(nextEdge));
			nextEdge = he.getOpposite(he.getNext(nextEdge));
			v2 = he.getTarget(he.getOpposite(nextEdge));
			f = he.getFace(nextEdge);
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for B: " << elapsed.count() << " s\n";
}

/**
* Compute GeodesicDistance
**/
void computeGeodesicDisctance() {
	std::cout << "Computing GeodesicDistance..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	SolverFinal.compute(Lc);
	GeodesicDistance = SolverFinal.solve(B);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for GeodesicDistance: " << elapsed.count() << " s\n";
}


// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	switch (key) {
	case '1':
	{
		MatrixXd C;
		igl::jet(GeodesicDistance, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '2':
	{
		setInitialTemperature();
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
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
		igl::jet(Temperature, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '6':
	{
		for (int i = 0; i < nbSteps; i++)
			computeTimeStepExplicit();
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
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
			igl::jet(Temperature, true, C); // Assign per-vertex colors
			viewer.data().set_colors(C); // Add per-vertex colors
			return false; };
		viewer.launch(); // run the editor
		return true;
	}
	case '8':
	{
		computeTimeStepImplicit();
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '9':
	{
		for (int i = 0; i < nbSteps; i++)
			computeTimeStepImplicit();
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
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
			igl::jet(Temperature, true, C); // Assign per-vertex colors
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

	auto totalstart = std::chrono::high_resolution_clock::now(); // for measuring time performances

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
	setInitialTemperature();
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
	std::cout << "Computing steps..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	for (int i = 0; i < nbStepsTotal; i++)
		computeTimeStepExplicit();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for " << nbStepsTotal << " steps: " << elapsed.count() << " s\n";
	computeNormalizedTemperatureGradient(he);
	computeB(he);
	computeGeodesicDisctance();
	setInitialTemperature();

	auto totalfinish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalelapsed = totalfinish - totalstart;
	std::cout << "Total computing time from the loading of the data to the obtention of the geodesic distance:" << totalelapsed.count() << " s\n";
	std::cout << "Including computing time for a total of " << nbStepsTotal << " steps of heat diffusion: " << elapsed.count() << " s\n";
	std::cout << "On a mesh of " << V.rows() << " points" << std::endl;

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
	igl::jet(Temperature,true,C); // Assign per-vertex colors
	viewer.data().set_colors(C); // Add per-vertex colors

	//viewer.core(0).align_camera_center(V, F); //not needed
	viewer.launch(); // run the viewer
}