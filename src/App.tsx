import React, { useContext, useEffect, useState } from "react";
import {
    ChakraProvider,
    Text,
    Heading,
    HStack,
    UnorderedList,
    ListItem,
    Divider,
    Icon,
    Button,
    Link,
} from "@chakra-ui/react";

// Theme
import customTheme from "./theme";

// Icons
import { SiTypescript, SiReact, SiChakraui } from "react-icons/si";
import { FaInfoCircle, FaTerminal, FaPlay } from "react-icons/fa";
import { BsPlayFill } from "react-icons/bs"
import { RepeatIcon } from "@chakra-ui/icons"

// Components
import ColorModeSwitcher from "./components/ColorModeSwitcher";
import SideBar from "./components/SideBar";
import Panel from "./components/Panel";
import Grid from "./components/Grid";
import Logo from "./components/Logo";
import AlgorithmMenu from "./components/AlgorithmMenu";
import SpeedMenu from "./components/SpeedMenu";

import { StoreContext } from "./Store";
import { StoreContextType } from "./@types/Store";

export const App: React.FC = () => {
    const store = useContext(StoreContext) as StoreContextType;
    const [addRandObs, setAddRandObs] = useState<boolean>(false);

    useEffect(() => {
        store.setCellSize(30);
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
            setAddRandObs(false);
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

                    {/* <Panel>
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
                    </Panel> */}

                    <Panel>
                        <HStack>
                            <Heading as="h2" size="md" variant="gradient">
                                Controls{" "}
                            </Heading>
                            <Icon as={FaTerminal} color="brand.blue.500" />
                        </HStack>
                        <ColorModeSwitcher />
                        {/* <HStack mt="2">
                            
                            
                        </HStack> */}
                        
                        <Text my="1" >Select Algorithm:</Text>
                        <AlgorithmMenu />
                        <Text my="1" >Select visualization speed:</Text>
                        <SpeedMenu />
                        {/* <Text my="1" >Obstacles:</Text> */}
                        
                        
                        {/* <Divider bg="gray.800"/> */}
                        <HStack mt="2">
                            <Button
                                onClick={() => {
                                    store.resetObstacles();
                                }}
                                
                                variant="brandPurple"
                                size="sm"
                            >
                                Erase Walls
                            </Button>
                            <Button 
                                onClick={() => setAddRandObs(true)}
                                
                                variant="brandPurple"
                                size="sm"
                            >
                                Random Walls
                            </Button>
                            <Button
                                onClick={() => {
                                    store.resetNodes();
                                    store.setIsStarted(false);
                                }}
                                
                                variant="brandPurple"
                                // rightIcon={<RepeatIcon />}
                                size="sm"
                                // w="100%"
                            >
                                Reset
                            </Button>
                        </HStack>

                        <Button
                            bgGradient="linear(to-l, brand.blue.500, brand.purple.500)"
                            onClick={() => store.setIsStarted(true)}
                            rightIcon={<BsPlayFill />}
                            mt="2"
                            width="100%"
                            color="white"
                        >
                            Visualize
                        </Button>
                        
                        
                    </Panel>
                </SideBar>

                <Grid />
            </HStack>
        </ChakraProvider>
    );
};
