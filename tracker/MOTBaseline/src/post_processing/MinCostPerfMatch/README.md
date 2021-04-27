# Algorithms for Maximum Cardinality Matching and Minimum Cost Perfect Matching Problems in General Graphs

I implemented these algorithms during my PhD, in 2011, following the description in:

*Gerards, A.M.H. (1995). Matching. In Ball, M., Magnanti, T., Monma, C., and Nemhauser, G., editors, Network Models, volume 7 of Handbooks in Operations Research and Management Science, chapter 3, pages 135-224. Elsevier.*

See Example.cpp for examples of how to use the API.

Compilation with G++:
```
g++ -O3 Example.cpp BinaryHeap.cpp Matching.cpp Graph.cpp -o example
```

## To use as a matching solver:
```
./example -f <filename> <--minweight | --max>
```
`--minweight` for minimum weight perfect matching

`--max` for maximum cardinality matching


**File format:**

the first two lines give n (number of vertices) and m (number of edges), respectively, followed by m lines, each with a tuple (u, v [, c]) representing the edges. In each tuple, u and v are the endpoints (0-based indexing) of the edge and c is its cost. The cost is optional if --max is specified.

Example, `input.txt`:
```
10
16
0 1 10
0 2 4
1 2 3
1 5 2
1 6 2
2 3 1
2 4 2
3 4 5
4 6 4
4 7 1
4 8 3
5 6 1
6 7 2
7 8 3
7 9 2
8 9 1
```

```
./example -f input.txt --minweight
```

Output:
```
Optimal matching cost: 14
Edges in the matching:
0 1
2 3
4 7
5 6
8 9
```

Feel free to contact me if you have any problem.
