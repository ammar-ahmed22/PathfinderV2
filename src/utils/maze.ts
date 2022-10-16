import { StoreContextType } from "../@types/Store";
import Vec2 from "../helpers/Vec2";
import { sleep } from "./async";

// All references to grid are mutating

export class MazeGenerator {
    public grid: boolean[][] = [];
    public stack: Vec2[] = [];
    public visited: Vec2[] = [];

    constructor(public dimensions: Vec2) {
        if (this.isEven(dimensions.x) || this.isEven(dimensions.y)) {
            throw new Error("dimensions must be odd!");
        }

        for (let row = 0; row < dimensions.y; row++) {
            this.grid[row] = [];
            for (let col = 0; col < dimensions.x; col++) {
                this.grid[row][col] = false;
            }
        }
        this.addOuterWalls();
        this.subdivide();
        this.stack.push(new Vec2(1, 1));
        this.visited.push(new Vec2(1, 1));
    }

    private addOuterWalls = () => {
        for (let row = 0; row < this.dimensions.y; row++) {
            if (row === 0 || row === this.dimensions.y - 1) {
                for (let col = 0; col < this.dimensions.x; col++) {
                    this.grid[row][col] = true;
                }
            } else {
                this.grid[row][0] = true;
                this.grid[row][this.dimensions.x - 1] = true;
            }
        }
    };

    private isEven = (num: number): boolean => num % 2 === 0;

    private subdivide = () => {
        for (let row = 1; row < this.dimensions.y - 1; row++) {
            if (this.isEven(row)) {
                for (let col = 1; col < this.dimensions.x - 1; col++) {
                    this.grid[row][col] = true;
                }
            } else {
                for (let col = 2; col < this.dimensions.x - 1; col += 2) {
                    this.grid[row][col] = true;
                }
            }
        }
    };

    private randIndex = (length: number) => Math.floor(Math.random() * length);

    private getNeighbours = (curr: Vec2): Vec2[] => {
        let neighbs: Vec2[] = [];

        const row = curr.y;
        const col = curr.x;

        // right
        if (col < this.dimensions.x - 2) {
            neighbs.push(new Vec2(col + 2, row));
        }
        // left
        if (col > 2) {
            neighbs.push(new Vec2(col - 2, row));
        }
        // top
        if (row > 2) {
            neighbs.push(new Vec2(col, row - 2));
        }
        // bottom
        if (row < this.dimensions.y - 2) {
            neighbs.push(new Vec2(col, row + 2));
        }

        return neighbs.filter((val) => !Vec2.ArrayIncludes(this.visited, val));
    };

    private removeWall = (curr: Vec2, chosen: Vec2) => {
        if (curr.x === chosen.x) {
            // same column
            const dy = chosen.y - curr.y;
            if (dy < 1) {
                // remove cell below chosen
                this.grid[chosen.y + 1][chosen.x] = false;
            } else {
                // remove cell above chosen
                this.grid[chosen.y - 1][chosen.x] = false;
            }
        } else {
            // same row
            const dx = chosen.x - curr.x;
            if (dx < 1) {
                // remove cell to right of chosen
                this.grid[chosen.y][chosen.x + 1] = false;
            } else {
                // remove cell to left of chosen
                this.grid[chosen.y][chosen.x - 1] = false;
            }
        }
    };

    private updateNodes = (store: StoreContextType) => {
        for (let row = 0; row < this.grid.length; row++) {
            for (let col = 0; col < this.grid[row].length; col++) {
                const obs: boolean = this.grid[row][col];
                const idx = new Vec2(col, row);

                store.updateNodeByIndex(idx, (prevNode) => {
                    prevNode.type = obs ? "obstacle" : "base";
                    prevNode.obstacle = obs;
                    return prevNode;
                });
            }
        }
    };

    private updateNodesAsync = async (
        store: StoreContextType,
        delay: number = 10
    ) => {
        for (let row = 0; row < this.grid.length; row++) {
            for (let col = 0; col < this.grid[row].length; col++) {
                const obs: boolean = this.grid[row][col];
                const idx = new Vec2(col, row);

                store.updateNodeByIndex(idx, (prevNode) => {
                    prevNode.type = obs ? "obstacle" : "base";
                    prevNode.obstacle = obs;
                    return prevNode;
                });
            }
            await sleep(delay);
        }
    };

    private shiftStartTarget = (
        store: StoreContextType,
        nodeType: "start" | "target"
    ) => {
        if (store.startIdx && store.targetIdx && !!store.nodes.length) {
            const idx: Vec2 =
                nodeType === "start" ? store.startIdx : store.targetIdx;
            const nodeToMove = store.nodes[idx.y][idx.x];
            let hasMoved = false;
            const neighbours = nodeToMove.getNeighbours(store.nodes, {
                allowDiagonals: true,
            });
            console.log({ moving: nodeType, nLength: neighbours.length });
            for (let i = 0; i < neighbours.length; i++) {
                const n = neighbours[i];
                const { index: nIndex } = n;
                if (!this.grid[nIndex.y][nIndex.x]) {
                    if (nodeType === "start") {
                        store.setStartIdx(nIndex);
                        hasMoved = true;
                    } else {
                        store.setTargetIdx(nIndex);
                        hasMoved = true;
                    }
                    break;
                }
            }
        }
    };

    private placeStartTarget = (store: StoreContextType) => {
        if (store.startIdx && store.targetIdx && !!store.nodes.length) {
            const s = store.startIdx;
            const t = store.targetIdx;

            let sHasMoved = false;
            let tHasMoved = false;

            if (this.grid[s.y][s.x]) {
                this.shiftStartTarget(store, "start");
                sHasMoved = true;
            }

            if (this.grid[t.y][t.x]) {
                this.shiftStartTarget(store, "target");
                tHasMoved = true;
            }

            if (!sHasMoved)
                store.updateNodeTypeByIndex(store.startIdx, "start");
            if (!tHasMoved)
                store.updateNodeTypeByIndex(store.targetIdx, "target");
        }
    };

    public animatedGeneration = async (store: StoreContextType) => {
        if (!!store.nodes.length) {
            // before generation
            await this.updateNodesAsync(store);

            while (!!this.stack.length) {
                const curr = this.stack.pop() as Vec2;
                const neighbours = this.getNeighbours(curr);
                if (!!neighbours.length) {
                    this.stack.push(curr);
                    const chosen =
                        neighbours[this.randIndex(neighbours.length)];
                    this.removeWall(curr, chosen);
                    this.updateNodes(store);
                    await sleep(0);
                    this.visited.push(chosen);
                    this.stack.push(chosen);
                }
            }
            // after generation
            this.placeStartTarget(store);
        }
    };

    public generate = (): boolean[][] => {
        while (!!this.stack.length) {
            //this.log();
            const curr = this.stack.pop() as Vec2;
            const neighbours = this.getNeighbours(curr);
            if (!!neighbours.length) {
                this.stack.push(curr);
                const chosen = neighbours[this.randIndex(neighbours.length)];
                this.removeWall(curr, chosen);
                this.visited.push(chosen);
                this.stack.push(chosen);
            }
        }

        return this.grid;
    };

    public log = (clear: boolean = false) => {
        let mazeString = "";
        this.grid.forEach((row) => {
            let rowString = "";
            row.forEach((val) => {
                rowString += val ? "⬛" : "⬜";
            });
            rowString += "\n";
            mazeString += rowString;
        });

        console.log(mazeString);
        if (clear) {
            console.clear();
        }
    };
}
