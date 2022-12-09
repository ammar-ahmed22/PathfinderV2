import React, { useRef, useEffect, useContext } from "react";
import { HStack } from "@chakra-ui/react";

// Components
import Panel from "./Panel";
import Cell from "./Cell";

// Utils
import { isCorner, animate } from "../utils/grid";

// Store
import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

// Helpers
import Vec2 from "../helpers/Vec2";
import { solvers } from "../helpers/algorithms";

const Grid: React.FC = () => {
  const gridRef = useRef<HTMLDivElement>();
  const store = useContext(StoreContext) as StoreContextType;

  // setting Target
  useEffect(() => {
    if (store.gridDim) {
      const halfHeight = Math.floor((store.gridDim.y - 1) / 2);
      store.setStartIdx(new Vec2(5, halfHeight));
      store.setTargetIdx(new Vec2(store.gridDim.x - 5, halfHeight));
    }
    // eslint-disable-next-line
  }, [store.gridDim]);

  // Settings initial grid dimensions
  useEffect(() => {
    if (gridRef.current && store.cellSize) {
      const { current } = gridRef;
      store.updateGridDimensions(current, store.cellSize);
    }
    // eslint-disable-next-line
  }, [gridRef, store.cellSize]);

  // Creating nodes after dimensions set
  useEffect(() => {
    if (store.gridDim && store.cellSize) {
      store.createNodes();
    }
    // eslint-disable-next-line
  }, [store.gridDim, store.cellSize]);

  // Resize event
  useEffect(() => {
    const resizeEventListener = () => {
      if (gridRef.current && store.cellSize) {
        const { current } = gridRef;
        store.updateGridDimensions(current, store.cellSize);
      }
    };

    window.addEventListener("resize", resizeEventListener);

    // return () => window.removeEventListener("resize", resizeEventListener);
    // eslint-disable-next-line
  }, []);

  useEffect(() => {
    if (
      store.nodes &&
      store.startIdx &&
      store.targetIdx &&
      store.status.started
    ) {
      const solver = solvers[store.selectedAlgorithm];
      solver.initialize({
        nodes: store.nodes,
        start: store.startIdx,
        target: store.targetIdx,
      });
      const start = performance.now();
      const path = solver.solve();
      const end = performance.now();

      const output = `Path found in: ${(end - start).toFixed(2)}ms`;
      store.addOutput(output);

      if (path) animate(store, path, solver.searched, store.visualDelay);
      if (path === undefined) {
        store.setFinished(true);
        store.addOutput("No path found!");
      }
    }
    // eslint-disable-next-line
  }, [store.status.started]);

  return (
    <Panel
      height="calc(100vh - var(--pf-space-5) * 2)"
      width="75vw"
      customRef={gridRef}
      styles={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
      }}
      id="grid"
    >
      {store.nodes &&
        !!store.nodes.length &&
        store.nodes.map((row, rowIdx) => {
          return (
            <HStack key={rowIdx} spacing="0">
              {row.map((node, colIdx) => {
                return (
                  <Cell
                    key={rowIdx + colIdx}
                    node={node}
                    corner={isCorner(
                      node.index,
                      store.nodes.length,
                      row.length
                    )}
                  />
                );
              })}
            </HStack>
          );
        })}
    </Panel>
  );
};

export default Grid;
