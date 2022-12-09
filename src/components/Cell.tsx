import React, { useContext, useState } from "react";
import { Box, Icon, keyframes, useColorModeValue } from "@chakra-ui/react";
import { motion } from "framer-motion";

// Icons
import { FaMapMarkerAlt, FaCrosshairs } from "react-icons/fa";

// Utils
import { borderRadii, createAnimation } from "../utils/cell";
import { getChakraCSSVar } from "../utils/colors";

// Helpers
import Vec2 from "../helpers/Vec2";

// Types
import { CellProps } from "../@types/components/Cell";

// Store
import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

const Cell: React.FC<CellProps> = ({ node, corner }) => {
  const borderSize = 1;

  const animBoxSize = node.size - 2 * borderSize;

  const animations = {
    visited: keyframes`
      0% { background-color: var(${getChakraCSSVar(
        "brand.purple.400"
      )}); height: ${animBoxSize / 4}px; width: ${
      animBoxSize / 4
    }px; border-radius: 100%; }
      100% { background-color: var(${getChakraCSSVar(
        "brand.blue.500"
      )}); height: ${animBoxSize}px; width: ${animBoxSize}px; border-radius: 0; }
    `,
    path: keyframes`
      0% { height: ${animBoxSize / 4}px; width: ${
      animBoxSize / 4
    }px; border-radius: 100%; }
      100% { height: ${animBoxSize}px; width: ${animBoxSize}px; border-radius: 0; }
    `,
  };

  const styleProps = {
    height: node.size + "px",
    width: node.size + "px",
    borderStyle: "solid",
    borderWidth: `${borderSize}px`,
    borderColor: useColorModeValue("gray.600", "gray.400"),
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  };

  const store = useContext(StoreContext) as StoreContextType;
  const [dragging, setDragging] = useState(false);

  const isStartTarget = node.type === "start" || node.type === "target";

  const handleDragStart = (ev: React.DragEvent<HTMLDivElement>) => {
    setDragging(true);
    ev.dataTransfer.setData("draggedNode", JSON.stringify(node));
    ev.dataTransfer.dropEffect = "move";
  };

  const handleDragOver = (ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = isStartTarget ? "none" : "move";
  };

  const handleDragDrop = (ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    const data = ev.dataTransfer.getData("draggedNode");
    const draggedNode = JSON.parse(data);

    if (draggedNode.type === "start") {
      store.setStartIdx(node.index);
    } else {
      store.setTargetIdx(node.index);
    }

    const draggedNodeIndex = new Vec2(draggedNode.index.x, draggedNode.index.y);
    store.updateNodeByIndex(draggedNodeIndex, (prevNode) => {
      prevNode.type = "base";
      return prevNode;
    });
    setDragging(false);
  };

  return (
    <Box
      {...styleProps}
      {...borderRadii(corner)}
      draggable={isStartTarget}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDrop={handleDragDrop}
      onMouseOver={(e) => {
        if (e.buttons === 1 && !isStartTarget && !dragging) {
          store.updateNodeByIndex(node.index, (prevNode) => {
            prevNode.obstacle = store.shiftPressed ? false : true;
            prevNode.type = store.shiftPressed ? "base" : "obstacle";
            return prevNode;
          });
        }
      }}
      onMouseDown={() => {
        if (!isStartTarget && !dragging) {
          store.updateNodeByIndex(node.index, (prevNode) => {
            prevNode.obstacle = store.shiftPressed ? false : true;
            prevNode.type = store.shiftPressed ? "base" : "obstacle";
            return prevNode;
          });
        }
      }}
      cursor={isStartTarget ? "grab" : "pointer"}
      className={`${node.type}-cell`}
    >
      {isStartTarget && (
        <Box
          height={animBoxSize + "px"}
          width={animBoxSize + "px"}
          display="flex"
          justifyContent="center"
          alignItems="center"
          bg={
            store.status.started
              ? node.type === "start"
                ? "path.start"
                : "path.end"
              : "none"
          }
        >
          <Icon
            as={node.type === "start" ? FaMapMarkerAlt : FaCrosshairs}
            color={store.status.started ? "white" : styleProps.borderColor}
          />
        </Box>
      )}
      {node.type === "visited" && (
        <Box
          as={motion.div}
          animation={createAnimation(animations.visited, 1, "ease-in")}
          height={animBoxSize + "px"}
          width={animBoxSize + "px"}
          bg="brand.blue.500"
          {...borderRadii(corner)}
        />
      )}
      {node.type === "path" && (
        <Box
          as={motion.div}
          animation={createAnimation(animations.path, 1, "ease-in")}
          height={animBoxSize + "px"}
          width={animBoxSize + "px"}
          bg={node.bg ? node.bg : "yellow.400"}
          {...borderRadii(corner)}
        />
      )}
      {node.type === "obstacle" && (
        <Box
          height={animBoxSize + "px"}
          width={animBoxSize + "px"}
          bg={styleProps.borderColor}
          {...borderRadii(corner)}
        />
      )}
    </Box>
  );
};

export default Cell;
