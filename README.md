# Pathfinder

This is a pathfinding algorithm visualizer I created as a learning exercise. The project and design is inspired by a similar [project](https://github.com/clementmihailescu/Pathfinding-Visualizer) created by Clement Mihailescu, founder of AlgoExpert.io; however, I did not look at his code to create my own variation. It was merely
used as design inspiration as I wanted to implement the algorithms and UI on my own to further my understanding of these concepts.

As I am quite well-versed in React, I decided to use this in conjunction with [ChakraUI](https://chakra-ui.com/) to make a standardized and robust UI. [Framer Motion](https://www.framer.com/motion/) was used for the animations. As this project is data structure and algorithm heavy, I decided to use TypeScript instead of JavaScript as statically typed languages help to mitigate errors.

I have created this project in the past, however, I was not happy with the design as well as the way the solving was done. So, this is the 
second iteration of the project with design updates as well as optimizations.

## Run Locally

#### Clone the project

```bash
  git clone https://github.com/ammar-ahmed22/PathfinderV2.git
```

#### Go to the project directory

```bash
  cd PathfinderV2
```

#### Install dependencies

```bash
  npm install
```
> with yarn: `yarn install`

#### Start the server

```bash
  npm start
```
> with yarn: `yarn start`

## Features

- Light/dark mode toggle
- Drag/drop/draw nodes (start, end, obstacles)
- Visualize and find paths using various pathfinding algorithms
- Generate maze (maybe?)
- Change visualization speed (fast, medium, slow + explicitly define delay between iterations)

### Algorithms
| Algorithm     | Description | Shortest Path Guaranteed? |
| ------------- | ----------- | -------------------------
| Djikstra's    | Uses distance from root (start) node to find shortest path. | Yes
| Greedy        | Uses heuristic (euclidean distance) to find path.           | No
| A*            | See Djikstra and Greedy above, follows same basic principle. Uses distance + plus heuristic (euclidean distance) to find shortest path.| Yes

### Abstractions 

| Abstraction     | Description | 
| ------------- | ----------- | 
| Node   | Used to store grid node data such as index in the grid, size, UI styles, node type (start, end, obstacle etc.). Also used as a base class to be extended for use in algorithms. i.e. distances and heuristic storing for A* etc. |
| Vec2            | 2D vector class used to simplify grid indices (stores column and row) as well as other miscellaneous 2D values such as grid dimensions, height and width of grid etc. Also implements useful methods such as euclidean distances between nodes etc. |
| Solver | Abstract base class for pathfinding algorithms. Implements solve method as well as searching queue, searched array and optimal path array. All algorithms extend this class. |
| Min Priority Queue | As described, heap based min priority queue for pathfinding algorithm optimization. Most algorithms require pulling the minimum value from a queue, min priority queue allows this in ~ O(logn) time instead of O(n) time if done with an array based queue.|

## Tech Stack

- React (TypeScript)
- [ChakraUI](https://chakra-ui.com/)
- [Framer Motion](https://www.framer.com/motion/)
