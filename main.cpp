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

// New for the project
MatrixXd X; // vector of vertices
MatrixXd CotAlpha; // Matrix of cot(alpha_ij)
MatrixXd CotBeta; // Matrix of cot(beta_ij)
MatrixXd L; // Laplace-Berltrami matrix
MatrixXd A; // Diagonal matrix of vertex area
MatrixXd D; // Length of edges

// Old from the TD 4 and useful

MatrixXd N_faces; // computed calling pre-defined functions of LibiGL
MatrixXd N_vertices; // computed calling pre-defined functions of LibiGL

MatrixXd lib_N_vertices; // computed using face-vertex structure of LibiGL
MatrixXi lib_Deg_vertices; // computed using face-vertex structure of LibiGL

MatrixXd he_N_vertices; // computed using the HalfEdge data structure

// Old from the TD 4 and useless

MatrixXd G_curvature; // computed calling pre-defined functions of LibiGL
MatrixXd he_G_curvature; // computed using the HalfEdge data structure


// New for the project

/**
* Compute X (he)
**/
void computeX(HalfedgeDS he) {
	X = MatrixXd::Zero(he.sizeOfVertices(), 3);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		X.row(i) = V.row(i);
	}
}

/**
* Compute CotAlpha, CotBeta and D (he)
**/
void computeAlphaBetaA(HalfedgeDS he) {
	std::cout << "Computing CotAlpha, CotBeta and D..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	CotAlpha = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	CotBeta = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	D = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
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
			CotAlpha(i, j) = 1 / tan(acos(w.dot(-v) / (v.norm() + w.norm()))); // angle at k
			CotBeta(i, k) = 1 / tan(acos(u.dot(-w) / (u.norm() + w.norm()))); // angle at j
			j = he.getTarget(he.getOpposite(pe));
			pe = he.getOpposite(he.getNext(pe));
			k = he.getTarget(he.getOpposite(pe));
		}
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for CotAlpha, CotBeta and D: " << elapsed.count() << " s\n";
}

/**
* Compute A (he)
**/
void computeA(HalfedgeDS he) {
	std::cout << "Computing A..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	A = MatrixXd::Zero(he.sizeOfVertices(), he.sizeOfVertices());
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		int vDCW = vertexDegreeCCW(he, i);
		int e = he.getEdge(i); // edge from x_j to x_i
		int j = he.getTarget(he.getOpposite(e)); // vertex before i
		for (int k = 0; k < vDCW; k++) {
			A(i, i) += D(i, j) * D(i, j) * (CotAlpha(i, j) + CotBeta(i, j));
			e = he.getOpposite(he.getNext(e));
			j = he.getTarget(he.getOpposite(e));
		}
		A(i, i) = 8 / A(i, i);
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
		A(i, i) = 8 / A(i, i);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for L: " << elapsed.count() << " s\n";
}

// Old from TD 4 and useful

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

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
	switch(key){
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
		default: break;
	}
	return false;
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

// Old from TD 4 and useless

/**
* Return the degree of a given vertex 'v'
* Exercice
**/
int vertexDegree(HalfedgeDS he, int v) {
	int vD = 0;
	for (int i = 0; i < he.sizeOfHalfedges(); i++) {
		if (he.getTarget(i) == v)
			vD++;
	}
	return vD;
}

/**
* Return the number of occurrence of vertex degrees: for d=3..n-1
* Exercice
**/
void vertexDegreeStatistics(HalfedgeDS he) {

	std::cout << "Computing vertex degree distribution using vertexDegree..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances
	int *vDS = new int[he.sizeOfVertices()];
	for (int i = 0; i < he.sizeOfVertices(); i++)
		vDS[i] = 0;
	for (int i = 0; i < he.sizeOfVertices(); i++)
		vDS[vertexDegree(he, i)]++;
	for (int i = 3; i < he.sizeOfVertices(); i++)
		if (vDS[i] != 0)
			std::cout << "number of degrees = " << i << " ; number of occurences = " << vDS[i] << std::endl;
	std::cout << "Zero for the other vertex degrees between d=3 and n-1" << std::endl;
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time using vertexDegree: " << elapsed.count() << " s\n";

	std::cout << "Computing vertex degree distribution using vertexDegreeCCW..." << std::endl;
	auto startCCW = std::chrono::high_resolution_clock::now(); // for measuring time performances
	int* vDSCCW = new int[he.sizeOfVertices()];
	for (int i = 0; i < he.sizeOfVertices(); i++)
		vDSCCW[i] = 0;
	for (int i = 0; i < he.sizeOfVertices(); i++)
		vDSCCW[vertexDegreeCCW(he, i)]++;
	for (int i = 3; i < he.sizeOfVertices(); i++)
		if (vDSCCW[i] != 0)
			std::cout << "number of degrees = " << i << " ; number of occurences = " << vDSCCW[i] << std::endl;
	std::cout << "Zero for the other vertex degrees between d=3 and n-1" << std::endl;
	auto finishCCW = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedCCW = finishCCW - startCCW;
	std::cout << "Computing time using vertexDegreeCCW: " << elapsedCCW.count() << " s\n";
}

// Compute lib_vertex degrees
/**
* (global, using libiGl data structure)
* Exercice
**/
void lib_vertexDegrees() {
	lib_Deg_vertices = MatrixXi::Zero(V.rows(), 1);
	for (int i = 0; i < F.rows(); i++) {
		lib_Deg_vertices(F.row(i)[0])++;
		lib_Deg_vertices(F.row(i)[1])++;
		lib_Deg_vertices(F.row(i)[2])++;
	}
	std::cout << "Computing vertex degree distribution using lib_vertexDegrees..." << std::endl;
	auto startlib = std::chrono::high_resolution_clock::now(); // for measuring time performances
	int* vDSlib = new int[V.rows()];
	for (int i = 0; i < V.rows(); i++)
		vDSlib[i] = 0;
	for (int i = 0; i < V.rows(); i++)
		vDSlib[lib_Deg_vertices(i)]++;
	for (int i = 3; i < V.rows(); i++)
		if (vDSlib[i] != 0)
			std::cout << "number of degrees = " << i << " ; number of occurences = " << vDSlib[i] << std::endl;
	std::cout << "Zero for the other vertex degrees between d=3 and n-1" << std::endl;
	auto finishlib = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedlib = finishlib - startlib;
	std::cout << "Computing time using lib_vertexDegrees: " << elapsedlib.count() << " s\n";
}
         
/**
* Return the number of boundaries of the mesh
* Exercice
**/
int countBoundaries(HalfedgeDS he){
	int c = 0;
	std::list<int> l = { };
	for (int i = 0; i < he.sizeOfHalfedges(); i++)
		if (he.getFace(i) == -1)
			l.push_back(i);
	while (!l.empty()) {
		c++;
		int e = l.front();
		l.pop_front();
		int pe = he.getOpposite(e);
		while (he.getFace(pe) != -1)
			pe = he.getOpposite(he.getNext(he.getNext(pe)));
		while (pe != e) {
			l.remove(pe);
			pe = he.getOpposite(pe);
			while (he.getFace(pe) != -1)
				pe = he.getOpposite(he.getNext(he.getNext(pe)));
		}
	}
	return c;
}

/**
* Compute the Gaussian curvatures (he)
* Exercice
**/
void gaussianCurvature(HalfedgeDS he) {
	std::cout << "Computing the Gaussian curvatures using gaussianCurvature..." << std::endl;
	auto starthe = std::chrono::high_resolution_clock::now(); // for measuring time performances
	he_G_curvature = MatrixXd::Zero(he.sizeOfVertices(), 1);
	double pi = 3.141592654;
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		double k = 2 * pi;
		double aVoronoi = 0;
		int vDCW = vertexDegreeCCW(he, i);
		MatrixXd alpha = MatrixXd::Zero(vDCW, 1);
		MatrixXd beta = MatrixXd::Zero(vDCW, 1);
		MatrixXd d = MatrixXd::Zero(vDCW, 1);
		int e = he.getEdge(i);
		int pe = he.getOpposite(he.getNext(e));
		int i0 = he.getTarget(he.getOpposite(e));
		int i1 = i;
		int i2 = he.getTarget(he.getOpposite(pe));
		for (int j = 0; j < vDCW; j++) {
			Vector3d u(V.row(i1)[0] - V.row(i0)[0], V.row(i1)[1] - V.row(i0)[1], V.row(i1)[2] - V.row(i0)[2]);
			Vector3d v(V.row(i2)[0] - V.row(i1)[0], V.row(i2)[1] - V.row(i1)[1], V.row(i2)[2] - V.row(i1)[2]);
			Vector3d w(V.row(i0)[0] - V.row(i2)[0], V.row(i0)[1] - V.row(i2)[1], V.row(i0)[2] - V.row(i2)[2]);
			d(j) = u.norm();
			alpha(j) = 1 / tan(acos(w.dot(-v) / (v.norm() + w.norm())));
			beta(j) = 1 / tan(acos(u.dot(-w) / (u.norm() + w.norm())));
			k -= acos(v.dot(-u) / (u.norm() + v.norm()));
			i0 = he.getTarget(he.getOpposite(pe));
			pe = he.getOpposite(he.getNext(pe));
			i2 = he.getTarget(he.getOpposite(pe));
		}
		for (int j = 0; j < vDCW; j++) {
			aVoronoi += d(j) * d(j) * (alpha(j) + beta((j - 1 + vDCW) % vDCW));
		}
		aVoronoi /= 8;
		he_G_curvature(i) = k / aVoronoi;
	}
	auto finishhe = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedhe = finishhe - starthe;
	std::cout << "Computing time using gaussianCurvature: " << elapsedhe.count() << " s\n";
}


// ------------ main program ----------------
int main(int argc, char *argv[]) {

//	if(argc<2) {
//		std::cout << "Error: input file required (.OFF)" << std::endl;
//		return 0;
//	}
//	std::cout << "reading input file: " << argv[1] << std::endl;

//	igl::readOFF(argv[1], V, F);
	igl::readOFF("../data/cat0.off",V,F);			// 2 boundaries
	//igl::readOFF("../data/chandelier.off", V, F);	// 10 boundaries
	//igl::readOFF("../data/cube_open.off", V, F);	// 1 boundary
	//igl::readOFF("../data/cube_tri.off", V, F);	// 0 boundary
	//igl::readOFF("../data/face.off", V, F);		// 1 boundary
	//igl::readOFF("../data/high_genus.off", V, F);	// 0 boundary
	//igl::readOFF("../data/homer.off", V, F);		// 1 boundary
	//igl::readOFF("../data/nefertiti.off", V, F);	// 1 boundary
	//igl::readOFF("../data/star.off", V, F);		// 0 boundary
	//igl::readOFF("../data/venus.off", V, F);		// 1 boundary

	//print the number of mesh elements
    std::cout << "Points: " << V.rows() << std::endl;

	HalfedgeBuilder* builder=new HalfedgeBuilder();

	HalfedgeDS he=builder->createMesh(V.rows(), F);

	// New for the project

	computeX(he);
	computeAlphaBetaA(he);
	computeA(he);
	computeL(he);

	// Old from TD 4 and useful

	// Compute normals

	// Compute per-face normals
	igl::per_face_normals(V,F,N_faces);

	// Compute per-vertex normals
	igl::per_vertex_normals(V,F,N_vertices);

	// Compute lib_per-vertex normals
	lib_vertexNormals();

	// Compute he_per-vertex normals
	vertexNormals(he);

	/*
	
	// Old from TD 4 and useless

	// print normals

	for (int i = 0; i < 15; i++) {
		if (i < V.rows()) {
			std::cout << " " << std::endl;
			std::cout << "Normal of vertex " << i << std::endl;
			std::cout << "From N_faces: " << N_faces.row(i) << std::endl;
			std::cout << "From N_vertices: " << N_vertices.row(i) << std::endl;
			std::cout << "From lib_N_vertices: " << lib_N_vertices.row(i) << std::endl;
			std::cout << "From he_N_vertices: " << he_N_vertices.row(i) << std::endl;
		}
	}

	// compute vertex degrees
	vertexDegreeStatistics(he);
	lib_vertexDegrees();

	// compute number of boundaries
	int B = countBoundaries(he);
	if (B <= 1)
		std::cout << "The mesh has " << B << " boundary" << std::endl;
	else
		std::cout << "The mesh has " << B << " boundaries" << std::endl;


	// Compute Gaussian curvatures

	// Compute G_curvature
	igl::gaussian_curvature(V, F, G_curvature);

	// Compute he_G_curvature
	gaussianCurvature(he);

	for (int i = 0; i < 15; i++) {
		if (i < V.rows()) {
			std::cout << " " << std::endl;
			std::cout << "Gaussian curvatures of vertex " << i << std::endl;
			std::cout << "From G_curvature: " << G_curvature.row(i) << std::endl;
			//std::cout << "From lib_G_curvature: " << lib_G_curvature.row(i) << std::endl;
			std::cout << "From he_G_curvature: " << he_G_curvature.row(i) << std::endl;
		}
	}

	*/

///////////////////////////////////////////

	// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V, F);
	viewer.data().set_normals(N_faces);
	std::cout<<
		"Press '1' for per-face normals calling pre-defined functions of LibiGL."<<std::endl<<
		"Press '2' for per-vertex normals calling pre-defined functions of LibiGL."<<std::endl<<
		"Press '3' for lib_per-vertex normals using face-vertex structure of LibiGL ."<<std::endl<<
		"Press '4' for HE_per-vertex normals using HalfEdge structure."<<std::endl;

	VectorXd Z;
	Z.setZero(V.rows(),1);

	// Z colors
    // Use the z coordinate as a scalar field over the surface

	for (int i = 0; i < V.rows(); i++)
		Z[i] = V.row(i)[2];

	MatrixXd C;
	igl::jet(Z,true,C); // Assign per-vertex colors
	viewer.data().set_colors(C); // Add per-vertex colors

	//viewer.core(0).align_camera_center(V, F); //not needed
	viewer.launch(); // run the viewer
}
