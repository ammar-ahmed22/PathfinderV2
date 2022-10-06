import React, { useContext, useEffect } from "react"
import {
  ChakraProvider,
  Text,
  Heading,
  HStack,
  UnorderedList,
  ListItem,
  Icon,
  Button
} from "@chakra-ui/react"

// Theme
import customTheme from "./theme";

// Icons
import { SiTypescript, SiReact } from "react-icons/si"
import { FaInfoCircle, FaTerminal } from "react-icons/fa";

// Components
import ColorModeSwitcher from "./components/ColorModeSwitcher";
import SideBar from "./components/SideBar";
import Panel from "./components/Panel";
import Grid from "./components/Grid";
import Logo from "./components/Logo";

import { StoreContext } from "./Store";
import { StoreContextType } from "./@types/Store";


export const App : React.FC = () => {
  const store = useContext(StoreContext) as StoreContextType;

  useEffect(() => {
    store.setCellSize(40);
  }, [])
  
  return (
  <ChakraProvider theme={customTheme}>
    
      <HStack spacing="5" padding="5" >
        <SideBar width="25vw" >
          <Panel bg="" styles={{ textAlign: "center", bgGradient: "linear(to-tr, brand.blue.500, brand.purple.500)", color: "white" }}>
            <HStack justify="center" align="center" >
              <Logo h="8" />
              <Heading as="h1" size="lg" >Pathfinder</Heading>
            </HStack>
            <Text fontSize="xs" mb="1" >Made with 🧠 by Ammar</Text>
            <Text fontSize="xs" >Built with <Icon as={SiReact} /> + <Icon as={SiTypescript} /> </Text>
          </Panel>

          <Panel>
            <HStack>
              <Heading as="h2" size="md" variant="gradient" >What is this? </Heading>
              <Icon as={FaInfoCircle} color="brand.blue.500"/>
            </HStack>
            <Text fontSize="sm" >
            This is a <Text as="span" color="brand.purple" fontWeight="bold" >pathfinding algorithm visualization</Text> tool I created as an interactive and 
            engaging way to learn more about algorithms. You can visualize the following algorithms (more coming soon):
            </Text>
            <UnorderedList fontSize="sm" pl="4" pt="2"> 
              <ListItem>A*</ListItem>
              <ListItem>Djikstra's</ListItem>
              <ListItem>Greedy Best First Search</ListItem>
            </UnorderedList>
          </Panel>

          <Panel>
            <HStack>
              <Heading as="h2" size="md" variant="gradient" >Controls </Heading>
              <Icon as={FaTerminal} color="brand.blue.500"/>
            </HStack>
            <ColorModeSwitcher />
            <Button
              colorScheme="brand.blue"
            >Start</Button>
          </Panel>

        </SideBar>

        <Grid />

      </HStack>
  </ChakraProvider>
  )
}
