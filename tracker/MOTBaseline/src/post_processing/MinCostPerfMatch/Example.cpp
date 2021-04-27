#include "Matching.h"
#include <fstream>
#include "Graph.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

pair< Graph, vector<double> > CreateRandomGraph()
{
	//random seed
	int x;
	cin >> x;
	srand( x );

	//Please see Graph.h for a description of the interface
	int n = 50;

	Graph G(n);
	vector<double> cost;
	for(int i = 0; i < n; i++)
		for(int j = i+1; j < n; j++)
			if(rand()%10 == 0)
			{
				G.AddEdge(i, j);
				cost.push_back(rand()%1000);
			}

	return make_pair(G, cost);
}

Graph ReadGraph(string filename)
{
	//Please see Graph.h for a description of the interface

	ifstream file;
	file.open(filename.c_str());

	string s;
	getline(file, s);
	stringstream ss(s);
	int n;
	ss >> n;
	getline(file, s);
	ss.str(s);
	ss.clear();
	int m;
	ss >> m;

	Graph G(n);
	for(int i = 0; i < m; i++)
	{
		getline(file, s);
		ss.str(s);
		ss.clear();
		int u, v;
		ss >> u >> v;

		G.AddEdge(u, v);
	}

	file.close();
	return G;
}

pair< Graph, vector<double> > ReadWeightedGraph(string filename)
{
	//Please see Graph.h for a description of the interface

	ifstream file;
	file.open(filename.c_str());

	string s;
	getline(file, s);
	stringstream ss(s);
	int n;
	ss >> n;
	getline(file, s);
	ss.str(s);
	ss.clear();
	int m;
	ss >> m;

	Graph G(n);
	vector<double> cost(m);
	for(int i = 0; i < m; i++)
	{
		getline(file, s);
		ss.str(s);
		ss.clear();
		int u, v;
		double c;
		ss >> u >> v >> c;

		G.AddEdge(u, v);
		cost[G.GetEdgeIndex(u, v)] = c;
	}

	file.close();
	return make_pair(G, cost);
}

void MinimumCostPerfectMatchingExample(string filename)
{
	Graph G;
	vector<double> cost;
	
	//Read the graph
	pair< Graph, vector<double> > p = ReadWeightedGraph(filename);
	//pair< Graph, vector<double> > p = CreateRandomGraph();
	G = p.first;
	cost = p.second;

	//Create a Matching instance passing the graph
	Matching M(G);

	//Pass the costs to solve the problem
	pair< list<int>, double > solution = M.SolveMinimumCostPerfectMatching(cost);

	list<int> matching = solution.first;
	double obj = solution.second;

	cout << "Optimal matching cost: " << obj << endl;
	cout << "Edges in the matching:" << endl;
	for(list<int>::iterator it = matching.begin(); it != matching.end(); it++)
	{
		pair<int, int> e = G.GetEdge( *it );

		cout << e.first << " " << e.second << endl;
	}
}

void MaximumMatchingExample(string filename)
{
	Graph G = ReadGraph(filename);
	Matching M(G);

	list<int> matching;
	matching = M.SolveMaximumMatching();

	//cout << "Number of edges in the maximum matching: " << matching.size() << endl;
	//cout << "Edges in the matching:" << endl;
	for(list<int>::iterator it = matching.begin(); it != matching.end(); it++)
	{
		pair<int, int> e = G.GetEdge( *it );

		cout << e.first << " " << e.second << endl;
	}
}

int main(int argc, char* argv[])
{
	string filename = "";
	string algorithm = "";

	int i = 1;
	while(i < argc)
	{
		string a(argv[i]);
		if(a == "-f")
			filename = argv[++i];
		else if(a == "--minweight")
			algorithm = "minweight";
		else if(a == "--max")
			algorithm = "max";
		i++;
	}

	if(filename == "" || algorithm == "")
	{
		cout << "usage: ./example -f <filename> <--minweight | --max>" << endl;
		cout << "--minweight for minimum weight perfect matching" << endl;
		cout << "--max for maximum cardinality matching" << endl;
		cout << "file format:" << endl;
		cout << "the first two lines give n (number of vertices) and m (number of edges)," << endl;
		cout << "followed by m lines, each with a tuple (u, v [, c]) representing the edges," << endl;
	   	cout << "where u and v are the endpoints (0-based indexing) of the edge and c is its cost" << endl;	
		cout << "the cost is optional if --max is specified" << endl;
		return 1;
	}

	try
	{
		if(algorithm == "minweight")
			MinimumCostPerfectMatchingExample(filename);
		else
			MaximumMatchingExample(filename);
	}
	catch(const char * msg)
	{
		cout << msg << endl;
		return 1;
	}

	return 0;
}



