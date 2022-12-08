import Vec2 from "../helpers/Vec2";

export const generateMaze = (grid: boolean[][], dimensions: Vec2) => {
  // Initialize grid with all empty
  for (let row = 0; row < dimensions.y; row++) {
    grid[row] = [];
    for (let col = 0; col < dimensions.x; col++) {
      grid[row][col] = false;
    }
  }

  // Add border walls
  addOuterWalls(grid, dimensions);

  // Recursive subdivision algorithm
  addInnerWalls(grid, {
    isHorizontal: true,
    minX: 1,
    maxX: dimensions.x - 2,
    minY: 1,
    maxY: dimensions.y - 2,
  });
};

const addOuterWalls = (grid: boolean[][], dimensions: Vec2) => {
  for (let row = 0; row < dimensions.y; row++) {
    if (row === 0 || row === dimensions.y - 1) {
      for (let col = 0; col < dimensions.x; col++) {
        grid[row][col] = true;
      }
    } else {
      grid[row][0] = true;
      grid[row][dimensions.x - 1] = true;
    }
  }
};

interface AddInnerWallsOpts {
  isHorizontal: boolean;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

const addInnerWalls = (
  grid: boolean[][],
  { isHorizontal, minX, maxX, minY, maxY }: AddInnerWallsOpts
) => {
  if (isHorizontal) {
    if (maxX - minX < 2) {
      return;
    }

    const y = Math.floor(randomNumber(minY, maxY) / 2) * 2;
    addHWall(grid, minX, maxX, y);

    addInnerWalls(grid, {
      isHorizontal: !isHorizontal,
      minX,
      maxX,
      minY,
      maxY: y - 1,
    });

    addInnerWalls(grid, {
      isHorizontal: !isHorizontal,
      minX,
      maxX,
      minY: y + 1,
      maxY,
    });
  } else {
    if (maxY - minY < 2) {
      return;
    }

    const x = Math.floor(randomNumber(minX, maxX) / 2) * 2;

    addVWall(grid, minY, maxY, x);

    addInnerWalls(grid, {
      isHorizontal: !isHorizontal,
      minX,
      maxX: x + 1,
      minY,
      maxY,
    });
    addInnerWalls(grid, {
      isHorizontal: !isHorizontal,
      minX: x + 1,
      maxX,
      minY,
      maxY,
    });
  }
};

const addHWall = (grid: boolean[][], minX: number, maxX: number, y: number) => {
  var hole = Math.floor(randomNumber(minX, maxX) / 2) * 2 + 1;

  for (let i = minX; i <= maxX; i++) {
    if (i === hole) grid[y][i] = false;
    else grid[y][i] = true;
  }
};

const addVWall = (grid: boolean[][], minY: number, maxY: number, x: number) => {
  let hole = Math.floor(randomNumber(minY, maxY) / 2) * 2 + 1;

  for (var i = minY; i <= maxY; i++) {
    if (i == hole) grid[i][x] = false;
    else grid[i][x] = true;
  }
};

const randomNumber = (min: number, max: number) => {
  return Math.floor(Math.random() * (max - min + 1) + min);
};

export const displayMaze = (grid: boolean[][]) => {
  let mazeString = "";
  grid.forEach((row) => {
    let rowString = "";
    row.forEach((val) => {
      rowString += val ? "⬛" : "⬜";
    });
    rowString += "\n";
    mazeString += rowString;
  });

  console.log(mazeString);
};
