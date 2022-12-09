import { StepType } from "@reactour/tour";
import {
  Box,
  Text,
  Flex,
  Button,
} from "@chakra-ui/react";

export const steps: StepType[] = [
  {
    selector: "#info-panel",
    position: "right",
    content: () => {
      return (
        <Box>
          <Text fontSize="lg" variant="gradient" textAlign="center">
            Hello!
          </Text>
          <Text textAlign="center">
            Welcome to{" "}
            <Text as="span" variant="gradient">
              Pathfinder
            </Text>
            .
          </Text>
        </Box>
      );
    },
  },
  {
    selector: "#ammar",
    content: () => {
      return (
        <Box>
          <Text mb="2">
            My name is{" "}
            <Text as="span" variant="gradient">
              Ammar Ahmed.
            </Text>{" "}
            I'm a student at the University of Waterloo.
          </Text>
          <Text>Check out my personal portfolio here.</Text>
        </Box>
      );
    },
  },
  {
    selector: "#info-panel",
    content: () => (
      <Box>
        <Text fontSize="lg" variant="gradient">
          What does that mean?
        </Text>
        <Text>
          Pathfinding algorithms explore routes between nodes (locations)
          through traversing neighbours based on some defined logic. They
          start at one node and explore until the destination is reached. Once
          the destination has been reached, they can find a path between the
          start and destination (target) nodes based on some defined logic.
        </Text>
      </Box>
    ),
  },
  {
    selector: "#info-panel",
    position: "right",
    content: () => (
      <Box>
        <Text fontSize="lg" variant="gradient">
          What does this do?
        </Text>
        <Text>
          As computers are very fast, in order to see how an algorithm works,
          we have to slow it down to visualize it. This website visualizes
          various pathfinding algorithms looking to find a path between a
          start and target node in the face of random or user-set obstacles.
        </Text>
      </Box>
    ),
  },
  {
    selector: "#info-panel",
    content: () => (
      <Box>
        <Text fontSize="lg" variant="gradient" mb="2">
          Which algorithms are visualized?
        </Text>
        <Text fontSize="md" variant="gradient">
          Djikstra's Algorithm
        </Text>
        <Text>
          Explores all the nodes and keeps track of the distances to each,
          choosing the shortest path at each step until it reaches the
          destination. This ensures that the final path it finds is the
          shortest possible.
        </Text>
        <Text fontWeight="bold" mb="2">
          Shortest path guaranteed
        </Text>
        <Text fontSize="md" variant="gradient">
          Greed Best First Algorithm
        </Text>
        <Text>
          Prioritizes paths that appear to be leading closer to the goal. It
          uses heuristics to estimate the distance to the goal and focuses on
          the most promising paths. It can be faster than other algorithms.
        </Text>
        <Text fontWeight="bold" mb="2">
          Shortest path NOT guaranteed
        </Text>
        <Text fontSize="md" variant="gradient">
          A* Search Algorithm
        </Text>
        <Text>
          Uses the actual distance traveled and an estimated distance to the
          goal to choose the most promising paths.
        </Text>
        <Text fontWeight="bold">Shortest path guaranteed</Text>
      </Box>
    ),
  },
  {
    selector: "#github",
    content: "You can read more and check out the code here.",
  },
  {
    selector: "#grid",
    position: [20, 20],
    content: "This is where all the magic happens.",
  },
  {
    selector: ".start-cell",
    content: () => (
      <Text>
        This is the{" "}
        <Text color="green" as="span" fontWeight="bold">
          start
        </Text>{" "}
        node.
      </Text>
    ),
  },
  {
    selector: ".target-cell",
    content: () => (
      <Text>
        This is the{" "}
        <Text color="red" as="span" fontWeight="bold">
          target
        </Text>{" "}
        node.
      </Text>
    ),
  },
  {
    selector: "#grid",
    position: [20, 20],
    content: () => {
      return (
        <Text>
          Drag and drop the{" "}
          <Text as="span" color="green" fontWeight="bold">
            start
          </Text>{" "}
          and{" "}
          <Text as="span" color="red" fontWeight="bold">
            target
          </Text>{" "}
          nodes to move them.
        </Text>
      );
    },
  },
  {
    selector: "#grid",
    position: [20, 20],
    content:
      "Click any empty grid square to draw a wall. You can also hold and drag to draw many at once.",
  },
  {
    selector: "#grid",
    position: [20, 20],
    content: "Hold down Shift and click/draw to erase walls.",
  },
  {
    selector: "#erase-walls",
    content: "You can also click here to erase all drawn walls.",
  },
  {
    selector: "#random-walls",
    content: "This will draw a random assortment of walls.",
  },
  {
    selector: "#maze-walls",
    content:
      "This will generate a maze of walls using recursive backtracking.",
  },
  {
    selector: ".algorithm-menu",
    content: () => {
      return (
        <Text>
          Here you can select the{" "}
          <Text as="span" variant="gradient" fontWeight="bold">
            algorithm
          </Text>{" "}
          you would like to visualize.
        </Text>
      );
    },
  },
  {
    selector: ".speed-menu",
    content:
      "Here you can select the speed at which you want to view the visualization.",
  },
  {
    selector: "#visualize",
    content: "Click here to start the visualization",
  },
  {
    selector: "#reset",
    content: "Click here to reset the grid and visualization.",
  },
  {
    selector: "#output-panel",
    content: "Visualazation metrics and information will be displayed here.",
  },
  {
    selector: "#legend-panel",
    content:
      "You can reference what the different styles for grid squares mean here.",
  },
  {
    selector: "#help",
    content: ({ setIsOpen, setCurrentStep }) => {
      return (
        <Box>
          <Text>Click here anytime to show this tutorial.</Text>
          <Flex justify="center">
            <Button
              variant="brandGradient"
              size="sm"
              onClick={() => {
                setIsOpen(false);
                setCurrentStep(0);
              }}
            >
              Got It
            </Button>
          </Flex>
        </Box>
      );
    },
  },
];