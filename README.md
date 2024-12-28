_Этот документ [доступен на русском языке](/README.ru.md)_

# Rust Recipes 🍳 [![build](https://github.com/char16t/rust-recipes/actions/workflows/build.yml/badge.svg)](https://github.com/char16t/rust-recipes/actions/workflows/build.yml) ![coverage](https://char16t.github.io/rust-recipes/badges/flat.svg)

## Motivation

This repository contains implementations of algorithms and some data structures in Rust. All of them are implemented without dependencies, using only the standard Rust library. They can be used as building blocks for implementing more complex algorithms or programs. In most cases, all implementations consist of simple syntactic constructs available in imperative programming languages, so they are relatively easy to rewrite in another programming language.

In today's world of software development, programs are overgrown with many layers of abstractions from frameworks and ready-made libraries, but the main work is done by the small code size and functionality of the kernel. Superfluous abstractions lead to the fact that as the computing power of computers grows, our programs consume more and more computing resources while doing the same things they did 10 years ago.

Small or structurally simple programs written from scratch are easier to understand, maintain, fix, and analyze their performance. They are also much less expensive in terms of the infrastructure used to run them.

Use this repository to look up an implementation of an algorithm, understand it better and use it in your project. All implementations are covered by tests, so you can be sure in which situations they will work (or write a test for your own case and quickly check).

## Table of Contents

 * Input and output (`input.rs`)
   * Reading one, two, three, four and a vector of numbers
   * Reading strings
 * Bitwise operations (`bits.rs`)
   * Bitwise operations for vectors
   * Bitwise operations for matrices
   * Set representation
   * Long arithmetic for bitwise operations
 * Sorting and search algorithms (`sort.rs`)
   * Binary search
 * Algorithms on graphs (`graphs.rs`)
   * Unweighted graphs based on adjacency list (directed and undirected)
     * Depth-first search
     * Breadth-first search
     * Checking graph connectivity
     * Topological sorting (Kahn's algorithm)
     * Finding cycles in a graph (Floyd's algorithm, two pointers)
     * Search for strong connectivity components (Kosaraju's algorithm)
     * Eulerian path for connected graphs
     * Edge non-intersecting paths
     * Vertex non-intersecting paths
   * Weighted graphs based on the adjacency list (directed and undirected)
     * Depth-first search
     * Breadth-first search
     * Checking graph connectivity
     * Finding the shortest path (Dijkstra's Algorithm)
     * Topological sorting (Kahn's algorithm)
     * Finding cycles in a graph (Floyd's algorithm, two pointers)
     * Minimum spanning tree (Kruskal's algorithm)
     * Maximum spanning tree (Kruskal's algorithm)
     * Search for strong connectivity components (Kosaraju's algorithm)
     * Eulerian path for connected graphs
     * Maximum flow (Ford–Fulkerson algorithm)
     * Minimum cut (Ford–Fulkerson algorithm)
     * Edge non-intersecting paths
     * Vertex non-intersecting paths
   * Weighted graphs based on the adjacency matrix (directed and undirected)
     * Finding the shortest path (Floyd–Warshall algorithm)
   * Unweighted graphs based on the adjacency matrix (directed and undirected)
   * Unweighted graphs based on a list of edges (directed and undirected)
   * Weighted graphs based on a list of edges (directed and undirected)
     * Finding the shortest path (Bellman-Ford algorithm)
   * Successor Graph
   * Solution to the 2-SAT problem
 * Range queries (`ranges.rs`)
   * Queries to static arrays
     * Requests for sum
       * Array of prefix sums
       * Array of prefix sums 2D
       * Array of prefix sums 3D
     * Requests for minimum
       * Sparse table
   * Tree structures
      * Binary index tree (Fenwick tree)
      * Segment tree
 * Sequence compression (`sequences.rs`)
 * Algorithms on trees (`trees.rs`)
   * Tree based on the adjacency list
     * Depth-first search
     * Breadth-first search
     * Diameter of a tree
   * Tree based on adjacency list + adaptation of successor graph for trees
     * Depth-first search
     * Breadth-first search
     * Diameter of a tree
     * Least common ancestor for two vertices
     * Distance between two vertices
   * Pruefer code (tree encoding and decoding)
 * Maths
   * Number theory (`numbers.rs`)
     * Checking that a number is prime
     * Decomposition of a number into prime factors
     * Eratosthenes' sieve
     * Greatest common divisor (GСD, Euclid's algorithm)
     * Extended Euclid's algorithm
     * Least common multiple
     * Fast exponentiation
     * Fast modular exponentiation
     * Calculating the modulo inverse (Euler's theorem)
     * Solving equations in integers (Diophantine equations)
     * Chinese remainder theorem
   * Combinatorics (`combinatorics.rs`)
     * Factorial
     * Binomial coefficient
     * Multinomial coefficient
     * Catalan's number
     * Counting the number of combinations
     * Counting the number of combinations with repetitions
     * Counting the number of placements
     * Counting the number of placements with repetitions
     * Counting the number of permutations
     * Counting the number of permutations with repetitions
     * All combinations
     * All combinations with repetitions
     * All placements
     * All placements with repetitions
     * All permutations
     * All permutations with repetitions
     * Size of set union (inclusions-exclusions)
     * Count of derangements (or "disorders")
     * Counting different combinations in such a way that symmetric combinations are counted only once (Burnside's lemma)
   * Matrices (`matrices.rs`)
     * Matrix transpose
     * Matrix addition (and subtraction)
     * Matrix multiplication
     * Multiplication of a matrix by a scalar
     * Fast matrix exponentiation
     * Linear recurrence relations
     * Fibonacci numbers
     * Number of paths in the graph containing exactly N edges
     * Length of the shortest path in a graph from a to b containing exactly n edges
     * Solving systems of linear equations (Gauss method)
   * Probability (`probability.rs`)
     * Discrete random variable
     * Mathematical expectation
     * Random number according to a given distribution
     * Markov chains
   * Game theory (`games.rs`)
     * Nim game
     * Misere game (the reverse of the Nim game)
     * Grandi's game (Sprague–Grundy theorem)
   * Fourier transform (`fourier.rs`)
     * Fast Fourier Transform
     * Multiplication of polynomials
     * Signal processing
   * Geometry (`geometry.rs`)
     * Complex numbers
     * Vector product
     * Checking the position of a point relative to a line
     * Distance from a point to a line
 * Algorithms for working with strings (`strings.rs`)
   * de Bruijn sequence
   * Trie
   * Largest common subsequence
   * Editorial distance (Levenshtein distance)
   * Jaro-Winkler similarity
   * Fuzzy search
     * Fuzzy search using Levenshtein distance
     * Fuzzy search using Levenshtein distance + synonyms
     * Fuzzy search using Jaro-Winkler similarity
     * Fuzzy search using Jaro-Winkler similarity + synonyms
   * Polynomial hashing
   * Pattern matching (search for all occurrences)
   * Counting the number of different substrings of length k
 * Pseudorandom number generators (`random.rs`)
   * Xoshiro256
 * Heaps (`heaps.rs`)
   * Binary min-heap
   * Binary max-heap
 * Concurrency (`concurrency.rs`)
   * Thread pool

## Build and test

Detailed instructions for building and testing are given in [/docs/Development.adoc](/docs/Development.adoc). Most of the build details are hidden in the [Makefile](/Makefile).

![Code coverage](/docs/codecov.png "Code coverage")
