import React, { useContext, useEffect, useState } from "react";
import {
    Text,
    Heading,
    HStack,
    Icon,
    Button,
    Link,
    useBreakpointValue,
    SimpleGrid,
    useMediaQuery,
    Modal,
    ModalOverlay,
    ModalHeader,
    ModalBody,
    ModalContent,
    ModalFooter,
    Box,
} from "@chakra-ui/react";

// Icons
import { SiTypescript, SiReact, SiChakraui } from "react-icons/si";
import {
    FaTerminal,
    FaClipboardList,
    FaMapMarkerAlt,
    FaCrosshairs,
} from "react-icons/fa";
import { BsPlayFill } from "react-icons/bs";
import { RepeatIcon } from "@chakra-ui/icons";

// Components
import ColorModeSwitcher from "./components/ColorModeSwitcher";
import SideBar from "./components/SideBar";
import Panel from "./components/Panel";
import Grid from "./components/Grid";
import Logo from "./components/Logo";
import AlgorithmMenu from "./components/AlgorithmMenu";
import SpeedMenu from "./components/SpeedMenu";
import CustomDivider from "./components/CustomDivider";
import LegendCell from "./components/LegendCell";
import type { LegendCellProps } from "./@types/components/LegendCell";
import Video from "./components/Video";

import Sneakpeek from "./assets/videos/SneakpeekPathfinder.mp4";

// Utils
import { createRandomObstacles } from "./utils/grid";
import { MazeGenerator } from "./utils/maze";

// Store
import { StoreContext } from "./Store";
import { StoreContextType } from "./@types/Store";

export const App: React.FC = () => {
    const store = useContext(StoreContext) as StoreContextType;
    const [addRandObs, setAddRandObs] = useState<boolean>(false);
    const [showModal, setShowModal] = useState<boolean>(false);

    useEffect(() => {
        store.setCellSize(30);
    }, []);

    useEffect(() => {
        if (addRandObs) {
            createRandomObstacles(store, 0.25);
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

    const generateMaze = async () => {
        if (store.gridDim) {
            const generator = new MazeGenerator(store.gridDim);
            await generator.animatedGeneration(store);
        }
    };

    const legendCellMapping: Omit<LegendCellProps, "size">[] = [
        {
            cellName: "Start",
            icon: FaMapMarkerAlt,
        },
        {
            cellName: "Target",
            icon: FaCrosshairs,
        },
        {
            cellName: "Visited",
            bg: "brand.blue.500",
        },
        {
            cellName: "Wall",
            bg: "borderColor",
        },
        {
            cellName: "Path",
            bgGradient: "linear(to-r, path.start, path.end)",
        },
    ];

    const legendCellSize = useBreakpointValue({
        base: "3vmin",
        lg: "5vmin"
    })

    const [isSmallerThan48em] = useMediaQuery("(max-width: 48em)");
    const [hIsSmallerThan30em] = useMediaQuery("(max-height: 30em)");

    useEffect(() => {
        if (isSmallerThan48em || hIsSmallerThan30em){
            setShowModal(true);
        } else {
            setShowModal(false);
        }
    }, [isSmallerThan48em, hIsSmallerThan30em])

    if (showModal) {
        return (
                <Modal isOpen={true} onClose={() => {}} isCentered size={{ base: "xs", md: "sm" }} >
                    <ModalOverlay backdropFilter="blur(10px)"/>
                    <ModalContent bgGradient="linear(to-tr, brand.blue.500, brand.purple.500)" color="white" >
                    <ModalHeader  >Larger Viewport Required </ModalHeader>
            
                    <ModalBody>
                        <Text mb="2" >
                            Please open this page on a device with a larger screen or increase your viewport size. 
                            Due to the nature of this application, smaller viewports are not supported.
                        </Text>
                        <Text mb="2">Sneakpeek:</Text>
                        <Video src={`${Sneakpeek}#t=0.1`} />
                    </ModalBody>
                    <ModalFooter />
                    </ModalContent>
                </Modal>
        )
    } else {
        return (
            <HStack spacing="5" padding="5">
                
                <SideBar width="25vw">
                    <Panel
                        bg=""
                        styles={{
                            textAlign: "center",
                            bgGradient:
                                "linear(to-tr, brand.blue.500, brand.purple.500)",
                            color: "white",
                            position: "relative",
                        }}
                    >
                        <HStack justify="center" align="center" mt="2">
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
                            <ColorModeSwitcher
                                position="absolute"
                                top="0"
                                left="0"
                                mt="2"
                                ml="2"
                            />
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

                    <Panel styles={{ position: "relative" }}>
                        <HStack>
                            <Heading as="h2" size="md" variant="gradient">
                                Controls{" "}
                            </Heading>
                            <Icon as={FaTerminal} color="brand.blue.500" />
                        </HStack>

                        <Heading my="1" as="h4" size="sm" variant="gradient">
                            Visualization
                        </Heading>
                        
                        <SimpleGrid my="2" columns={{ base: 1, lg: 2 }} spacing={2} >
                            <AlgorithmMenu />
                            <SpeedMenu />
                        </SimpleGrid>

                        <Heading my="1" as="h4" size="sm" variant="gradient">
                            Walls
                        </Heading>
                        <HStack my="2">
                            <Button
                                onClick={() => {
                                    store.resetObstacles();
                                }}
                                variant="brandPurple"
                                size="sm"
                            >
                                Erase
                            </Button>
                            <Button
                                onClick={() => setAddRandObs(true)}
                                variant="brandPurple"
                                size="sm"
                            >
                                Random
                            </Button>
                            <Button
                                onClick={() => generateMaze()}
                                variant="brandPurple"
                                size="sm"
                            >
                                Maze
                            </Button>
                        </HStack>

                        <CustomDivider my="2" />

                        <Button
                            onClick={() => {
                                store.resetNodes();
                                store.setIsStarted(false);
                            }}
                            variant="brandPurple"
                            rightIcon={<RepeatIcon />}
                            size="sm"
                            w="100%"
                        >
                            Reset
                        </Button>
                        <Button
                            onClick={() => store.setIsStarted(true)}
                            rightIcon={<BsPlayFill />}
                            mt="2"
                            size="sm"
                            width="100%"
                            variant="brandGradient"
                        >
                            Visualize
                        </Button>
                    </Panel>

                    <Panel>
                        <HStack>
                            <Heading as="h2" size="md" variant="gradient">
                                Legend{" "}
                            </Heading>
                            <Icon as={FaClipboardList} color="brand.blue.500" />
                        </HStack>

                        <SimpleGrid my="2" width="100%" columns={{ base: 3, lg: 5 }} spacing="2">
                            {legendCellSize && legendCellMapping.map((legendCellProps) => {
                                return (
                                    <LegendCell
                                        size={legendCellSize}
                                        borderWidth="1px"
                                        {...legendCellProps}
                                    />
                                );
                            })}
                        </SimpleGrid>
                    </Panel>
                </SideBar>

                <Grid />
            </HStack>
    );
    }
    
};
