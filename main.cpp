/************************************************************************************************************/

// The parts that can be changed are indicated between this kind of lines
// They are related to the sources, the number of steps, and the mesh,
// and located at the very start of the code, before the definitions of functions,
// and in the main function.

/************************************************************************************************************/


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

using namespace Eigen; // To use the classes provided by Eigen library
using namespace std;

MatrixXd V;
MatrixXi F;
MatrixXd N_vertices; // Computed calling pre-defined functions of LibiGL


/************************************************************************************************************/
//
// Parameters you can change

const int nbSources = 1; // Number of sources of heat
int Sources[nbSources] = { 0 }; // Sources of heat

// Choose this version if you want to see the anisotropia on the cube mesh :
/*
const int nbSources = 1; // Number of sources of heat
int Sources[nbSources] = { 9 }; // Sources of heat
*/
/************************************************************************************************************/


///////////////////////////////////////////
//
// For the heat method


/************************************************************************************************************/
//
// Parameter you can change

int nbSteps = 1000; // Number of time steps between animations
int nbMaxSteps = 40000; // Maximum number of time steps

/************************************************************************************************************/

MatrixXd GeodesicDistance; // Final distance function

MatrixXd Temperature; // Temperature, called U in the paper
MatrixXd NormalizedTemperatureGradient; // Normalized gradient of Temperature, called X in the paper
MatrixXd CenterOfFaces; // Centers of the faces
MatrixXd B; // Integrated divergences of NormalizedTemperatureGradient

SparseMatrix<double> CotAlpha; // Matrix of cot(alpha_ij)
SparseMatrix<double> CotBeta; // Matrix of cot(beta_ij)
SparseMatrix<double> Distances; // Length of edges

double dt; // Time step
double h; // Mean spacing between adjacent nodes
MatrixXd A; // Vector of vertex areas; can be voronoi or barycentric areas
SparseMatrix<double> Lc; // Laplace-Berltrami matrix, but only cotan operator
SparseMatrix<double> SparseMatrixExplicit;
SparseMatrix<double> LeftSideImplicit;
SimplicialCholesky<SparseMatrix<double>> SolverImplicit;
SimplicialCholesky<SparseMatrix<double>> SolverFinal;

///////////////////////////////////////////
//
// For the Dijkstra-based method

VectorXd DijkstraDistances; // Computed using Dijkstra algorithm

///////////////////////////////////////////
//
// Useful when the mesh is the sphere

MatrixXd TheoreticalShereDistance; // Computed using the theoretical formula

///////////////////////////////////////////
 

///////////////////////////////////////////
// General

/**
* Rescale the mesh in [0,1]^3
**/
void rescale() {
	std::cout << "Rescaling..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

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
* Initialise Temperature matrix
**/
void setInitialTemperature() {
	Temperature = MatrixXd::Zero(V.rows(), 1);
	for (int i = 0; i < nbSources; i++)
		Temperature(Sources[i]) = 1;
}

///////////////////////////////////////////
//
// For the heat method

/**
* Compute CotAlpha, CotBeta, Distances and dt
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
			stackCotAlpha.push_back(Eigen::Triplet<double>(i, j, 1 / tan(acos(ei.dot(-ej) / (ej.norm() * ei.norm()))))); // angle at k
			stackCotBeta.push_back(Eigen::Triplet<double>(i, k, 1 / tan(acos(ei.dot(-ek) / (ek.norm() * ei.norm()))))); // angle at j
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
	if (Distances.nonZeros() == 0) h = 0;
	else h = Distances.sum() / Distances.nonZeros();
}

/**
* Compute A (Voronoï version)
**/
void computeVoronoiArea(HalfedgeDS he) {
	std::cout << "Computing A (Voronoï)..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

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
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for A (Voronoï): " << elapsed.count() << " s\n";
}

/**
* Compute A
**/
void computeBarycentricArea(HalfedgeDS he) {
	std::cout << "Computing A..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	A = MatrixXd::Zero(he.sizeOfVertices(), 1);
	for (int i = 0; i < he.sizeOfVertices(); i++) {
		double vertexArea = 0;
		int edge1 = he.getEdge(i);
		int edge2 = he.getOpposite(he.getNext(edge1));
		do {
			int j = he.getTarget(he.getOpposite(edge1));
			int k = he.getTarget(he.getOpposite(edge2));
			double lenij = (V.row(i) - V.row(j)).norm();
			double lenik = (V.row(i) - V.row(k)).norm();
			double lenjk = (V.row(j) - V.row(k)).norm();
			double halfPerimeter = (lenij + lenik + lenjk) / 2;
			vertexArea += sqrt(halfPerimeter * (halfPerimeter - lenij) * (halfPerimeter - lenik) * (halfPerimeter - lenjk)) / 3;
			edge1 = edge2;
			edge2 = he.getOpposite(he.getNext(edge2));
		} while (edge1 != he.getEdge(i));
		A(i) = vertexArea;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for A: " << elapsed.count() << " s\n";
}

/**
* Compute Lc
**/
void computeL(HalfedgeDS he) {
	std::cout << "Computing Lc..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

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
	std::cout << "Computing time for Lc: " << elapsed.count() << " s\n";
}

/**
* Compute SparseMatrixExplicit
**/
void computeSparseMatrixExplicit() {
	std::cout << "Computing SparseMatrixExplicit..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

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
	auto start = std::chrono::high_resolution_clock::now();

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
* Compute NormalizedTemperatureGradient and EdgesNormalizedTemperatureGradient
**/
void computeNormalizedTemperatureGradient(HalfedgeDS he) {
	std::cout << "Computing NormalizedTemperatureGradient and CenterOfFaces..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	
	NormalizedTemperatureGradient = MatrixXd::Zero(F.rows(), 3);
	CenterOfFaces = MatrixXd::Zero(F.rows(), 3);
	for (int f = 0; f < F.rows(); f++) {
		int e = he.getEdgeInFace(f);
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
		CenterOfFaces.row(f) = (V.row(i) + V.row(j) + V.row(k)) / 3;
	}
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for NormalizedTemperatureGradient and CenterOfFaces: " << elapsed.count() << " s\n";
}

/**
* Compute B
**/
void computeB(HalfedgeDS he) {
	std::cout << "Computing B..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	
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
void computeGeodesicDistance() {
	std::cout << "Computing GeodesicDistance..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	SolverFinal.compute(Lc);
	GeodesicDistance = SolverFinal.solve(B);
	GeodesicDistance *= -1;
	GeodesicDistance  -= GeodesicDistance(Sources[0]) * MatrixXd::Ones(V.rows(), 1);

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for GeodesicDistance: " << elapsed.count() << " s\n";
}

///////////////////////////////////////////
//
// For the Dijstra-based method

/*
 Computes the Dijkstra distances to the sources
 Parameters:
	 - he: the HalfEdge data structure
	 - sources: the list of sources, given as their indices in he / V
	 - nbSources: the number of sources
 Returns:
	- the matrix of distances
*/
void ComputeDijkstra(HalfedgeDS he) {
	std::cout << "Computing Dijkstra method..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); // for measuring time performances

	DijkstraDistances = VectorXd::Constant(he.sizeOfVertices(), std::numeric_limits<double>::infinity());
	for (int i = 0; i < nbSources; i++) DijkstraDistances(Sources[i]) = 0;
	int* queue = new int[he.sizeOfHalfedges()];
	int head = 0; // The head of the queue : vertices to be processed
	int tail = 0; // The tail of the queue : the last inserted vertex
	for (int i = 0; i < nbSources; i++) {
		DijkstraDistances(Sources[i]) = 0;
		queue[tail] = Sources[i];
		tail++;
	}
	while (head != tail) {
		int v = queue[head]; // vertex to be processed
		head++;
		int edge = he.getEdge(v); // edge pointing towards v
		do {
			int neigbour = he.getTarget(he.getOpposite(edge));
			double edgeLength = (V.row(neigbour) - V.row(v)).norm();
			if (DijkstraDistances(neigbour) > DijkstraDistances(v) + edgeLength) {
				DijkstraDistances(neigbour) = DijkstraDistances(v) + edgeLength;
				queue[tail] = neigbour;
				tail++;
			}
			edge = he.getOpposite(he.getNext(edge));
		} while (edge != he.getEdge(v));
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for Dijkstra method: " << elapsed.count() << " s\n";
}


///////////////////////////////////////////
//
// Useful when the mesh is the sphere

/**
* Compute TheoreticalShereDistance
**/
void TheoreticalSphereDistance() {
	MatrixXd D = MatrixXd::Zero(V.rows(), V.rows());
	MatrixXd center = MatrixXd::Constant(1, 3, 0.5);
	for (int i = 0; i < V.rows(); i++) {
		for (int j = 0; j < V.rows(); j++) {
			Vector3d vi = (V.row(i) - center).normalized();
			Vector3d vj = (V.row(j) - center).normalized();
			D(i, j) = acos(vi.dot(vj)) / 2;
		}
	}
	TheoreticalShereDistance = D.col(0);
}

/**
* Print Mean Absolute Error (MAE) on the sphere
**/
void MAESphere() {
	GeodesicDistance *= TheoreticalShereDistance.maxCoeff() / GeodesicDistance.maxCoeff();
	float hm = 0.;
	float dijk = 0;
	for (int i = 0; i < V.rows(); i++) {
		hm += abs(GeodesicDistance(i) - TheoreticalShereDistance(i));
		dijk += abs(DijkstraDistances(i) - TheoreticalShereDistance(i));
	}
	hm /= V.rows();
	dijk /= V.rows();
	std::cout << "MAE on the sphere via the heat method: " << hm << std::endl;
	std::cout << "MAE on the sphere via the Dijkstra-based method: " << dijk << std::endl;
}

///////////////////////////////////////////
//
// Useful to have a cube mesh of 10 vertices per side 

void Cube(MatrixXd& V, MatrixXi& F) {
	int c = 10;
	std::map<int, int> OriginalToNew;
	std::map<int, int> NewToOriginal;
	MatrixXd V1 = MatrixXd::Zero(6 * c * c - 12 * c + 8, 3);
	MatrixXi F1 = MatrixXi::Zero(6 * (c - 1) * (c - 1) * 2, 3);
	int l = 0;
	for (int i = 0; i < c; i++)
		for (int j = 0; j < c; j++)
			for (int k = 0; k < c; k++)
			{
				if (i == 0 || i == c - 1 || j == 0 || j == c - 1 || k == 0 || k == c - 1)
				{
					int m = i + 10 * j + 100 * k;
					OriginalToNew[m] = l;
					NewToOriginal[l] = m;
					V1.row(l) << i, j, k;
					l++;
				}
			}
	l = 0;
	// Face devant à droite
	for (int k = 1; k < c; k++)
		for (int i = 1; i < c; i++)
		{
			F1.row(l) << OriginalToNew[i + 100 * k], OriginalToNew[i - 1 + 100 * k], OriginalToNew[i - 1 + 100 * (k - 1)];
			l++;
			F1.row(l) << OriginalToNew[i + 100 * k], OriginalToNew[i - 1 + 100 * (k - 1)], OriginalToNew[i + 100 * (k - 1)];
			l++;
		}
	// Face devant à gauche
	for (int k = 1; k < c; k++)
		for (int j = 1; j < c; j++)
		{
			F1.row(l) << OriginalToNew[10 * j + 100 * k], OriginalToNew[10 * (j - 1) + 100 * (k - 1)], OriginalToNew[10 * (j - 1) + 100 * k];
			l++;
			F1.row(l) << OriginalToNew[10 * j + 100 * k], OriginalToNew[10 * j + 100 * (k - 1)], OriginalToNew[10 * (j - 1) + 100 * (k - 1)];
			l++;
		}
	// Face derrière à droite
	for (int k = 1; k < c; k++)
		for (int j = 1; j < c; j++)
		{
			F1.row(l) << OriginalToNew[(c - 1) + 10 * j + 100 * k], OriginalToNew[(c - 1) + 10 * (j - 1) + 100 * k], OriginalToNew[(c - 1) + 10 * (j - 1) + 100 * (k - 1)];
			l++;
			F1.row(l) << OriginalToNew[(c - 1) + 10 * j + 100 * k], OriginalToNew[(c - 1) + 10 * (j - 1) + 100 * (k - 1)], OriginalToNew[(c - 1) + 10 * j + 100 * (k - 1)];
			l++;
		}
	// Face derrière à gauche
	for (int k = 1; k < c; k++)
		for (int i = 1; i < c; i++)
		{
			F1.row(l) << OriginalToNew[i + 10 * (c - 1) + 100 * k], OriginalToNew[i - 1 + 10 * (c - 1) + 100 * (k - 1)], OriginalToNew[i - 1 + 10 * (c - 1) + 100 * k];
			l++;
			F1.row(l) << OriginalToNew[i + 10 * (c - 1) + 100 * k], OriginalToNew[i + 10 * (c - 1) + 100 * (k - 1)], OriginalToNew[i - 1 + 10 * (c - 1) + 100 * (k - 1)];
			l++;
		}
	// Face en bas
	for (int j = 1; j < c; j++)
		for (int i = 1; i < c; i++)
		{
			F1.row(l) << OriginalToNew[i + 10 * j], OriginalToNew[i - 1 + 10 * (j - 1)], OriginalToNew[i - 1 + 10 * j];
			l++;
			F1.row(l) << OriginalToNew[i + 10 * j], OriginalToNew[i + 10 * (j - 1)], OriginalToNew[i - 1 + 10 * (j - 1)];
			l++;
		}
	// Face en haut
	for (int j = 1; j < c; j++)
		for (int i = 1; i < c; i++)
		{
			F1.row(l) << OriginalToNew[i + 10 * j + 100 * (c - 1)], OriginalToNew[i - 1 + 10 * j + 100 * (c - 1)], OriginalToNew[i - 1 + 10 * (j - 1) + 100 * (c - 1)];
			l++;
			F1.row(l) << OriginalToNew[i + 10 * j + 100 * (c - 1)], OriginalToNew[i - 1 + 10 * (j - 1) + 100 * (c - 1)], OriginalToNew[i + 10 * (j - 1) + 100 * (c - 1)];
			l++;
		}
	V = V1;
	F = F1;
}

///////////////////////////////////////////
//
// Useful if you want to subdivise naively a mesh

void Subdivise(MatrixXd& V, MatrixXi& F) {
	MatrixXd V1 = MatrixXd::Zero(V.rows() + F.rows(), 3);
	MatrixXi F1 = MatrixXi::Zero(3 * F.rows(), 3);
	for (int i = 0; i < V.rows(); i++)
		V1.row(i) = V.row(i);
	for (int i = 0; i < F.rows(); i++) {
		int j = V.rows() + i;
		V1.row(j) = (V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3;
		F1.row(3 * i + 0) << F(i, 0), F(i, 1), j;
		F1.row(3 * i + 1) << F(i, 1), F(i, 2), j;
		F1.row(3 * i + 2) << F(i, 2), F(i, 0), j;
	}
	V = V1;
	F = F1;
}

///////////////////////////////////////////
//
// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	switch (key) {
	case 'R':
	{
		MatrixXd C;
		igl::jet(B, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case 'S':
	{
		MatrixXd C;
		igl::jet(-TheoreticalShereDistance, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case 'G':
	{
		for (int f = 0; f < F.rows(); f++) {
			viewer.data().add_edges(
				CenterOfFaces.row(f),
				CenterOfFaces.row(f) + NormalizedTemperatureGradient.row(f) / 8,
				Eigen::RowVector3d(1, 1, 1));
		}
		return true;
	}
	case '1':
	{
		MatrixXd C;
		igl::jet(-GeodesicDistance, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '2':
	{
		MatrixXd C;
		igl::jet(-DijkstraDistances, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '3':
	{
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
	case '4':
	{
		setInitialTemperature();
		MatrixXd C;
		igl::jet(Temperature, true, C); // Assign per-vertex colors
		viewer.data().set_colors(C); // Add per-vertex colors
		return true;
	}
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
		viewer.data().set_normals(N_vertices);
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
		viewer.data().set_normals(N_vertices);
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

///////////////////////////////////////////


///////////////////////////////////////////

// ------------ main program ----------------
int main(int argc, char* argv[]) {


	/************************************************************************************************************/
	//
	// Load the mesh of your choice

	//igl::readOFF(argv[1], V, F);

	// Meshes without a boundary
	igl::readOFF("../data/bunny.off", V, F);		// 0 boundary
	//igl::readOFF("../data/sphere.off", V, F);		// 0 boundary
	//igl::readOFF("../data/twisted.off", V, F);	// 0 boundary
	//igl::readOFF("../data/output.off", V, F);		// 0 boundary
	//igl::readOFF("../data/high_genus.off", V, F);	// 0 boundary
	//igl::readOFF("../data/letter_a.off", V, F);	// 0 boundary

	// Meshes without a boundary but with too few points ; it should not work
	//igl::readOFF("../data/cube_tri.off", V, F);	// 0 boundary
	//igl::readOFF("../data/star.off", V, F);		// 0 boundary

	// Meshes with at least one boundary ; it does not work, but you can try out of curiosity
	//igl::readOFF("../data/cube_open.off", V, F);	// 1 boundary
	//igl::readOFF("../data/nefertiti.off", V, F);	// 1 boundary
	//igl::readOFF("../data/face.off", V, F);		// 1 boundary
	//igl::readOFF("../data/homer.off", V, F);		// 1 boundary
	//igl::readOFF("../data/venus.off", V, F);		// 1 boundary
	//igl::readOFF("../data/cat0.off",V,F);			// 2 boundaries
	//igl::readOFF("../data/chandelier.off", V, F);	// 10 boundaries

	// Meshes with unknown properties
	//igl::readOFF("../data/egea.off", V, F);
	//igl::readOFF("../data/gargoyle_tri.off", V, F);
	//igl::readOFF("../data/icosahedron.off", V, F);
	//igl::readOFF("../data/octagon.off", V, F);
	//igl::readOFF("../data/tetrahedron.off", V, F);
	//igl::readOFF("../data/torus_33.off", V, F);
	/************************************************************************************************************/


	/************************************************************************************************************/
	//
	// Uncomment if you want to replace the OFF mesh with a cube
	 
	//Cube(V, F);
	/************************************************************************************************************/


	/************************************************************************************************************/
	//
	// Uncomment if you want to subdivise naively the mesh
	 
	//Subdivise(V, F);
	/************************************************************************************************************/


	//print the number of mesh elements
	std::cout << "Points: " << V.rows() << std::endl;

	HalfedgeBuilder* builder = new HalfedgeBuilder();
	HalfedgeDS he = builder->createMeshWithFaces(V.rows(), F);

	rescale();
	
	// Heat method
	auto totalstart = std::chrono::high_resolution_clock::now(); // for measuring time performances
	setInitialTemperature();
	computeAlphaBetaDdt(he);
	std::cout << "dt: " << dt << std::endl;
	computeh();
	std::cout << "h: " << h << std::endl;
	computeBarycentricArea(he);
	computeL(he);
	computeSparseMatrixExplicit();
	computeSolverImplicit();
	std::cout << "Computing steps..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	MatrixXd OldTemperature = Temperature;
	int nbStepsComputed = 1;
	/************************************************************************************************************/
	computeTimeStepExplicit();
	//computeTimeStepImplicit();
	/************************************************************************************************************/
	while (Temperature(Sources[0]) == Temperature.maxCoeff() && nbStepsComputed < nbMaxSteps)
		{
			OldTemperature = Temperature;
			/************************************************************************************************************/
			computeTimeStepExplicit();
			//computeTimeStepImplicit();
			/************************************************************************************************************/
			nbStepsComputed++;
		}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Computing time for " << nbStepsComputed << " steps: " << elapsed.count() << " s\n";
	computeNormalizedTemperatureGradient(he);
	computeB(he);
	computeGeodesicDistance();
	setInitialTemperature();
	auto totalfinish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalelapsed = totalfinish - totalstart;
	std::cout << "Total computing time from the heat method:" << totalelapsed.count() << " s\n";
	std::cout << "Including computing time for a total of " << nbStepsComputed << " steps of heat diffusion: " << elapsed.count() << " s\n";
	std::cout << "On a mesh of " << V.rows() << " points" << std::endl;

	// Dijkstra method
	ComputeDijkstra(he);

	/************************************************************************************************************/
	//
	// To comment if the mesh is not the sphere

	/*
	// Theoretical sphere distance
	TheoreticalSphereDistance();

	// MAE on the sphere
	MAESphere();
	*/
	/************************************************************************************************************/


	// Compute per-vertex normals
	igl::per_vertex_normals(V, F, N_vertices);

	// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V, F);
	viewer.data().set_normals(N_vertices);
	std::cout <<
		"Press '1' to display the geodesic distance obtained via the heat method" << std::endl <<
		"Press '2' to display the distance obtained via the Dijkstra method" << std::endl <<
		"Press '3' to display the current temperature" << std::endl <<
		"Press '4' to reset and display the temperature" << std::endl <<
		"Press '5' to compute one step (explicit)" << std::endl <<
		"Press '6' to compute " << nbSteps << " steps (explicit)" << std::endl <<
		"Press '7' to animate (explicit)" << std::endl <<
		"Press '8' to compute one step (implicit)" << std::endl <<
		"Press '9' to compute " << nbSteps << " steps (implicit)" << std::endl <<
		"Press '0' to animate (implicit)" << std::endl <<
		"Press 'l' to display or mask the edges" << std::endl <<
		"Press 'r' to display the divergences of integrated temperature gradient" << std::endl <<
		"Press 's' to display the theoretical sphere distance (only if the mesh is the sphere)" << std::endl <<
		"Press 'g' to display the directions of the gradient of temperature" << std::endl;

	MatrixXd C;
	igl::jet(Temperature, true, C); // Assign per-vertex colors
	viewer.data().set_colors(C); // Add per-vertex colors

	//viewer.core(0).align_camera_center(V, F); //not needed
	viewer.launch(); // run the viewer
}