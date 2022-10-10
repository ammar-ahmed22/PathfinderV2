<p align="center">
    <img width="30"  alt="Pathfinder Website Logo" src="./public/favicons/apple-touch-icon.png?raw=true">
</p>
<h1 align="center">
  Pathfinder
</h1>

## What is this ❓

This is a pathfinding algorithm visualizer I created as a learning exercise. The project and design is inspired by a similar [project](https://github.com/clementmihailescu/Pathfinding-Visualizer) created by Clement Mihailescu, founder of AlgoExpert.io; however, I did not look at his code to create my own variation. It was merely
used as design inspiration as I wanted to implement the algorithms and UI on my own to further my understanding of these concepts.

As I am quite well-versed in React, I decided to use this in conjunction with [ChakraUI](https://chakra-ui.com/) to make a standardized and robust UI. [Framer Motion](https://www.framer.com/motion/) was used for the animations. As this project is data structure and algorithm heavy, I decided to use TypeScript instead of JavaScript as statically typed languages help to mitigate errors.

I have created this project in the past, however, I was not happy with the design as well as the way the solving was done. So, this is the 
second iteration of the project with design updates as well as optimizations.

Notable updates/optimizations in this design iteration include:
- Use of a min-priority queue for algorithms instead of an array-based queue
- Previously, animation was done through an asynchronous callback on each iteration of the algorithm, now, the algorithm runs synchronously with the animations done asynchronously (for visualization)
- Significantly more appealing UI/UX (in my opinion)

## Run Locally 💻

**Clone the project**

```bash
  git clone https://github.com/ammar-ahmed22/PathfinderV2.git
```

**Go to the project directory**

```bash
  cd PathfinderV2
```

**Install dependencies**

```bash
  npm install
```
> with yarn: `yarn install`

**Start the server**

```bash
  npm start
```
> with yarn: `yarn start`

## Tech Stack 👨‍💻

- React (TypeScript)
- [ChakraUI](https://chakra-ui.com/)
- [Framer Motion](https://www.framer.com/motion/)

## Features ✨

- Light/dark mode toggle
- Drag/drop/draw nodes (start, end, obstacles)
- Visualize and find paths using various pathfinding algorithms
- Generate maze (maybe?)
- Change visualization speed (fast, medium, slow + explicitly define delay between iterations)

## Algorithms 🧮
| Algorithm     | Description | Shortest Path Guaranteed? |
| ------------- | ----------- | -------------------------
| Djikstra's    | Uses distance from root (start) node to find shortest path. | Yes
| Greedy        | Uses heuristic (euclidean distance) to find path.           | No
| A*            | See Djikstra and Greedy above, follows same basic principle. Uses distance + plus heuristic (euclidean distance) to find shortest path.| Yes

## Abstractions ⚒️

| Abstraction     | Description | 
| ------------- | ----------- | 
| Node   | Used to store grid node data such as index in the grid, size, UI styles, node type (start, end, obstacle etc.). Also used as a base class to be extended for use in algorithms. i.e. distances and heuristic storing for A* etc. |
| Vec2            | 2D vector class used to simplify grid indices (stores column and row) as well as other miscellaneous 2D values such as grid dimensions, height and width of grid etc. Also implements useful methods such as euclidean distances between nodes etc. |
| Solver | Abstract base class for pathfinding algorithms. Implements solve method as well as searching queue, searched array and optimal path array. All algorithms extend this class. |
| Min Priority Queue | As described, heap based min priority queue for pathfinding algorithm optimization. Most algorithms require pulling the minimum value from a queue, min priority queue allows this in ~ O(logn) time instead of O(n) time if done with an array based queue.|

## CI/CD 🚗
| Name | Description |
| ---- | ------------|
| Format | GitHub action which formats all code with Prettier and then commits the changes on pushes to the main branch |

## References 📝
- [A* Search](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Djikstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Common Pathfinding Algorithms (helped with Greedy)](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [BFS for pathfinding](https://www.youtube.com/watch?v=KiCBXu4P-2Y)

