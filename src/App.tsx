import * as React from "react"
import {
  ChakraProvider,
  Text,
  Heading,
  HStack,
  UnorderedList,
  ListItem
} from "@chakra-ui/react"
import ColorModeSwitcher from "./components/ColorModeSwitcher";
import customTheme from "./theme";
import SideBar from "./components/SideBar";
import Panel from "./components/Panel";
import Board from "./components/Board";


export const App : React.FC = () => (
  <ChakraProvider theme={customTheme}>
    
      <HStack spacing="5" padding="5" >
        <SideBar width="25vw" >
          <Panel bg="" styles={{ textAlign: "center", bgGradient: "linear(to-tl, brand.blue, brand.purple)", color: "white" }}>
            <Heading as="h1" size="lg" >Pathfinder</Heading>
            <Text fontSize="xs" >Made with 🧠 by Ammar</Text>
          </Panel>

          <Panel>
            <Heading as="h2" size="md" variant="gradient" >What is this?</Heading>
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
            <Heading as="h2" size="md" variant="gradient" >Controls</Heading>
            <ColorModeSwitcher />
          </Panel>

        </SideBar>

        <Board />

      </HStack>
  </ChakraProvider>
)
