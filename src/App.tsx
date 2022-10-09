import React, { useContext, useEffect, useState } from "react";
import {
    ChakraProvider,
    Text,
    Heading,
    HStack,
    UnorderedList,
    ListItem,
    Icon,
    Button,
    Link,
} from "@chakra-ui/react";

// Theme
import customTheme from "./theme";

// Icons
import { SiTypescript, SiReact, SiChakraui } from "react-icons/si";
import { FaInfoCircle, FaTerminal } from "react-icons/fa";

// Components
import ColorModeSwitcher from "./components/ColorModeSwitcher";
import SideBar from "./components/SideBar";
import Panel from "./components/Panel";
import Grid from "./components/Grid";
import Logo from "./components/Logo";
import AlgorithmMenu from "./components/AlgorithmMenu";

import { StoreContext } from "./Store";
import { StoreContextType } from "./@types/Store";

export const App: React.FC = () => {
    const store = useContext(StoreContext) as StoreContextType;
    const [addRandObs, setAddRandObs] = useState<boolean>(false);

    useEffect(() => {
        store.setCellSize(40);
    }, []);

    useEffect(() => {
        if (addRandObs) {
            for (let row = 0; row < store.nodes.length; row++) {
                for (let col = 0; col < store.nodes[row].length; col++) {
                    const { index } = store.nodes[row][col];
                    if (
                        store.startIdx &&
                        store.targetIdx &&
                        !(
                            store.startIdx.equals(index) ||
                            store.targetIdx.equals(index)
                        )
                    ) {
                        store.updateNodeByIndex(index, (prevNode) => {
                            const obs: boolean = Math.random() < 0.25;
                            prevNode.obstacle = obs;
                            prevNode.type = obs ? "obstacle" : "base";
                            return prevNode;
                        });
                    }
                }
            }
        }
    }, [addRandObs]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Shift") {
                store.setShiftPressed(true);
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            if (e.key === "Shift") {
                store.setShiftPressed(false);
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        window.addEventListener("keyup", handleKeyUp);

        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            window.removeEventListener("keyup", handleKeyUp);
        };
    }, []);

    return (
        <ChakraProvider theme={customTheme}>
            <HStack spacing="5" padding="5">
                <SideBar width="25vw">
                    <Panel
                        bg=""
                        styles={{
                            textAlign: "center",
                            bgGradient:
                                "linear(to-tr, brand.blue.500, brand.purple.500)",
                            color: "white",
                        }}
                    >
                        <HStack justify="center" align="center">
                            <Logo h="8" />
                            <Heading as="h1" size="lg">
                                Pathfinder
                            </Heading>
                        </HStack>
                        <Text fontSize="xs" mb="1">
                            Made with 🧠 by Ammar
                        </Text>
                        <Text fontSize="xs">
                            Built with{" "}
                            <Link
                                variant="icon"
                                borderBottomColor="white"
                                href="https://reactjs.org/"
                                isExternal
                            >
                                <Icon as={SiReact} />
                            </Link>{" "}
                            +{" "}
                            <Link
                                variant="icon"
                                borderBottomColor="white"
                                href="https://www.typescriptlang.org/"
                                isExternal
                            >
                                <Icon as={SiTypescript} />
                            </Link>{" "}
                            +{" "}
                            <Link
                                variant="icon"
                                borderBottomColor="white"
                                href="https://chakra-ui.com/"
                                isExternal
                            >
                                <Icon as={SiChakraui} />
                            </Link>{" "}
                        </Text>
                    </Panel>

                    <Panel>
                        <HStack>
                            <Heading as="h2" size="md" variant="gradient">
                                What is this?
                            </Heading>
                            <Icon as={FaInfoCircle} color="brand.blue.500" />
                        </HStack>
                        <Text fontSize="sm">
                            This is a{" "}
                            <Text
                                as="span"
                                color="brand.purple"
                                fontWeight="bold"
                            >
                                pathfinding algorithm visualization
                            </Text>{" "}
                            tool I created as an interactive and engaging way to
                            learn more about algorithms. You can visualize the
                            following algorithms (more coming soon):
                        </Text>
                        <UnorderedList fontSize="sm" pl="4" pt="2">
                            <ListItem>A*</ListItem>
                            <ListItem>Djikstra's</ListItem>
                            <ListItem>Greedy Best First Search</ListItem>
                        </UnorderedList>
                    </Panel>

                    <Panel>
                        <HStack>
                            <Heading as="h2" size="md" variant="gradient">
                                Controls{" "}
                            </Heading>
                            <Icon as={FaTerminal} color="brand.blue.500" />
                        </HStack>
                        <ColorModeSwitcher />
                        <Button
                            colorScheme="brand.blue"
                            onClick={() =>
                                store.setIsStarted(
                                    true
                                )
                            }
                        >
                            Start
                        </Button>
                        <AlgorithmMenu />
                        <Button onClick={() => setAddRandObs(true)}>
                            Add Random Obstacles
                        </Button>
                        <Button onClick={() => {
                            store.resetNodes();
                            store.setIsStarted(false);
                        }}>
                            Reset
                        </Button>
                        <Button onClick={() => {
                            store.resetObstacles();
                        }}>
                            Erase Obstacles
                        </Button>
                    </Panel>
                </SideBar>

                <Grid />
            </HStack>
        </ChakraProvider>
    );
};
